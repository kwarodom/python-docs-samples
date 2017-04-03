#!/usr/bin/python
# -- coding: utf-8 --

# Copyright (C) 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sample that streams audio to the Google Cloud Speech API via GRPC."""

from __future__ import division

import contextlib
import functools
import re
import signal
import sys


import google.auth
import google.auth.transport.grpc
import google.auth.transport.requests
from google.cloud.proto.speech.v1beta1 import cloud_speech_pb2
from google.rpc import code_pb2
import grpc
import pyaudio
from six.moves import queue

import requests
import json
import time

import logging
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os.path
import uuid

from tornado.options import define, options

define("port", default=8888, help="run on the given port", type=int)

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/chatsocket", ChatSocketHandler),
        ]
        settings = dict(
            cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
        )
        super(Application, self).__init__(handlers, **settings)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", messages=ChatSocketHandler.cache)

class ChatSocketHandler(tornado.websocket.WebSocketHandler):
    waiters = set()
    cache = []
    cache_size = 200

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self):
        ChatSocketHandler.waiters.add(self)

    def on_close(self):
        ChatSocketHandler.waiters.remove(self)

    @classmethod
    def update_cache(cls, chat):
        cls.cache.append(chat)
        if len(cls.cache) > cls.cache_size:
            cls.cache = cls.cache[-cls.cache_size:]

    @classmethod
    def send_updates(cls, chat):
        print("sending updates")
        logging.info("sending message to %d waiters", len(cls.waiters))
        for waiter in cls.waiters:
            try:
                waiter.write_message(chat)
            except:
                logging.error("Error sending message", exc_info=True)

    def on_message(self, message):
        logging.info("got message %r", message)
        parsed = tornado.escape.json_decode(message)
        chat = {
            "id": str(uuid.uuid4()),
            "body": parsed["body"],
            }
        chat["html"] = tornado.escape.to_basestring(
            self.render_string("message.html", message=chat))

        ChatSocketHandler.update_cache(chat)
        ChatSocketHandler.send_updates(chat)

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# The Speech API has a streaming limit of 60 seconds of audio*, so keep the
# connection alive for that long, plus some more to give the API time to figure
# out the transcription.
# * https://g.co/cloud/speech/limits#content
DEADLINE_SECS = 60 * 3 + 5
SPEECH_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'

def make_channel(host, port):
    """Creates a secure channel with auth credentials from the environment."""
    # Grab application default credentials from the environment
    credentials, _ = google.auth.default(scopes=[SPEECH_SCOPE])

    # Create a secure channel using the credentials.
    http_request = google.auth.transport.requests.Request()
    target = '{}:{}'.format(host, port)

    return google.auth.transport.grpc.secure_authorized_channel(
        credentials, http_request, target)

def _audio_data_generator(buff):
    """A generator that yields all available data in the given buffer.

    Args:
        buff - a Queue object, where each element is a chunk of data.
    Yields:
        A chunk of data that is the aggregate of all chunks of data in `buff`.
        The function will block until at least one data chunk is available.
    """
    stop = False
    while not stop:
        # Use a blocking get() to ensure there's at least one chunk of data.
        data = [buff.get()]

        # Now consume whatever other data's still buffered.
        while True:
            try:
                data.append(buff.get(block=False))
            except queue.Empty:
                break

        # `None` in the buffer signals that the audio stream is closed. Yield
        # the final bit of the buffer and exit the loop.
        if None in data:
            stop = True
            data.remove(None)

        yield b''.join(data)

def _fill_buffer(buff, in_data, frame_count, time_info, status_flags):
    """Continuously collect data from the audio stream, into the buffer."""
    buff.put(in_data)
    return None, pyaudio.paContinue

# [START audio_stream]
@contextlib.contextmanager
def record_audio(rate, chunk):
    """Opens a recording stream in a context manager."""
    # Create a thread-safe buffer of audio data
    buff = queue.Queue()

    audio_interface = pyaudio.PyAudio()
    audio_stream = audio_interface.open(
        format=pyaudio.paInt16,
        # The API currently only supports 1-channel (mono) audio
        # https://goo.gl/z757pE
        channels=1, rate=rate,
        input=True, frames_per_buffer=chunk,
        # Run the audio stream asynchronously to fill the buffer object.
        # This is necessary so that the input device's buffer doesn't overflow
        # while the calling thread makes network requests, etc.
        stream_callback=functools.partial(_fill_buffer, buff),
    )

    yield _audio_data_generator(buff)

    audio_stream.stop_stream()
    audio_stream.close()
    # Signal the _audio_data_generator to finish
    buff.put(None)
    audio_interface.terminate()
# [END audio_stream]

def request_stream(data_stream, rate, interim_results=True):
    """Yields `StreamingRecognizeRequest`s constructed from a recording audio
    stream.

    Args:
        data_stream: A generator that yields raw audio data to send.
        rate: The sampling rate in hertz.
        interim_results: Whether to return intermediate results, before the
            transcription is finalized.
    """
    # The initial request must contain metadata about the stream, so the
    # server knows how to interpret it.
    recognition_config = cloud_speech_pb2.RecognitionConfig(
        # There are a bunch of config options you can specify. See
        # https://goo.gl/KPZn97 for the full list.
        encoding='LINEAR16',  # raw 16-bit signed LE samples
        sample_rate=rate,  # the rate in hertz
        # See http://g.co/cloud/speech/docs/languages
        # for a list of supported languages.
        language_code='th-TH',  # a BCP-47 language tag
        # language_code='en-EN',  # a BCP-47 language tag
    )
    streaming_config = cloud_speech_pb2.StreamingRecognitionConfig(
        interim_results=interim_results,
        config=recognition_config,
    )

    yield cloud_speech_pb2.StreamingRecognizeRequest(
        streaming_config=streaming_config)

    for data in data_stream:
        # Subsequent requests can all just have the content
        yield cloud_speech_pb2.StreamingRecognizeRequest(audio_content=data)

def listen_print_loop(recognize_stream):
    """Iterates through server responses and prints them.

    The recognize_stream passed is a generator that will block until a response
    is provided by the server. When the transcription response comes, print it.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    try:
        for resp in recognize_stream:
            if resp.error.code != code_pb2.OK:
                raise RuntimeError('Server error: ' + resp.error.message)

            if not resp.results:
                continue

            # Display the top transcription
            result = resp.results[0]
            transcript = result.alternatives[0].transcript

            # Display interim results, but with a carriage return at the end of the
            # line, so subsequent lines will overwrite them.
            #
            # If the previous result was longer than this one, we need to print
            # some extra spaces to overwrite the previous result
            overwrite_chars = ' ' * max(0, num_chars_printed - len(transcript))
            if not result.is_final:
                sys.stdout.write(transcript + overwrite_chars + '\r')
                sys.stdout.flush()

                num_chars_printed = len(transcript)
                print(transcript)
            else:
                print(transcript + overwrite_chars)

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if re.search(r'\b(exit|quit)\b', transcript, re.I):
                    print('Exiting..')
                    break
                elif re.search(r'\b(ออก)\b', transcript, re.I):
                    print('Exiting..')
                    break
                # elif transcript == u'ออโต้ เปิดไฟ ' or transcript == u'เอาโต เปิดไฟ ' or transcript == u'oppo เปิดไฟ ' \
                #         or transcript == u'auto เปิดไฟ ' or transcript == u' auto เปิดไฟ ' or transcript == u' oppo เปิดไฟ ':
                #     print('turning on light')
                # elif transcript == u'ออโต้ ช่วยด้วย ' or transcript == u'เอาโต ช่วยด้วย ' or transcript == u'oppo ช่วยด้วย ' \
                #         or transcript == u'auto ช่วยด้วย ' or transcript == u' auto ช่วยด้วย ' or transcript == u'เอาตัวช่วยด้วย ' \
                #         or transcript == u'auto ชื่อด้วย ' or transcript == u'auto ชื่อด้วย ' or transcript == u' oppo ช่วยด้วย ':
                #     print(u'พร้อมมาช่วยแล้วครับ')
                # elif transcript == u'ออโต้ เรียกตำรวจ ' or transcript == u'เอาโต เรียกตำรวจ ' or transcript == u'oppo เรียกตำรวจ ' \
                #         or transcript == u'auto เรียกตำรวจ ' or transcript == u' auto เรียกตำรวจ ' or transcript == u'เอาตัวเรียกตำรวจ ' \
                #         or transcript == u'auto เรียนตำรวจ ' or transcript == u'auto ตำรวจ ' or transcript == u'ขอโทษตำรวจ ' \
                #         or transcript == u'เอาตูดตำรวจ ' or transcript == u' auto ตำรวจ ' or transcript == u'auto ตำรวจ ':
                #     print(u'ตำรวจอยู่ระหว่างทางกำลังมา')
                # elif transcript == u'ออโต้ โรงพยาบาล ' or transcript == u'อัลโตโรงพยาบาล ' or transcript == u'oppo โรงพยาบาล ' \
                #         or transcript == u'auto โรงพยาบาล ' or transcript == u' auto โรงพยาบาล ' or transcript == u'เอาตัวโรงพยาบาล ' \
                #         or transcript == u'เอาตูดโรงพยาบาล ' or transcript == u'also โรงพยาบาล ':
                #     print(u'โรงพยาบาลกำลังมาช่วยเหลือครับ ')
                # elif transcript == u'ออโต้ เปิดแอร์ ' or transcript == u'เอาโต เปิดแอร์ ' or transcript == u'oppo เปิดแอร์ ' \
                #         or transcript == u'auto เปิดแอร์ ' or transcript == u' auto เปิดแอร์ ' or transcript == u'อัลโต้เปิดแอร์ ':
                #     print('turning on AC')
                # elif transcript == u'ออโต้ ไฟใหม้ ' or transcript == u'เอาโต ไฟไหม้ ' or transcript == u'oppo ไฟไหม้' \
                #         or transcript == u'auto ไฟไหม้ ' or transcript == u' auto ไฟไหม้ ' or transcript == u'รถตู้ไฟไหม้ '\
                #         or transcript == u'oppo find ไม่ ' or transcript == u'เอาตู้ไฟไหม้ ':
                #     print('ตำรวจดับเพลิงกำลังมาครับ')
                # elif transcript == u'ออโต้ เปิดทีวี ' or transcript == u'เอาโต เปิดทีวี ' or transcript == u'oppo เปิดทีวี ' \
                #         or transcript == u'auto เปิดทีวี ' or transcript == u' auto เปิดทีวี ':
                #     print('turning on TV')

                elif u'เปิดไฟ' in transcript:
                    turn_on_light()
                    print('turning on light')
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้ทำการเปิดไฟให้แล้วค่ะ  ')
                elif u'ปิดไฟ' in transcript:
                    turn_off_light()
                    print('turning off light')
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้ทำการปิดไฟให้แล้วค่ะ  ')
                elif u'ช่วยด้วย' in transcript:
                    print(u'พร้อมมาช่วยแล้วครับ')
                    text_to_speech(u'อัลโต้ได้ส่งข้อความและโทรเรียกคนในบ้านให้แล้วค่ะ ')
                elif u'เรียกตำรวจ' in transcript or u'ตำรวจ' in transcript:
                    print(u'ตำรวจอยู่ระหว่างทางกำลังมา')
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้เรียกตำรวจให้แล้วนะค่ะ อยู่ระหว่างทางค่ะ ')
                elif u'โรงพยาบาล' in transcript:
                    print(u'โรงพยาบาลกำลังมาช่วยเหลือครับ')
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้เรียกรถพยาบาลให้แล้วนะค่ะ อยู่ระหว่างทางค่ะ ')
                elif u'เปิดแอร์' in transcript:
                    turn_on_ac()
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้ทำการเปิดแอร์ให้แล้วค่ะ รอสักพักนะค่ะ ')
                    print('turning on AC')
                elif u'ปิดแอร์' in transcript:
                    turn_off_ac()
                    print('turning off AC')
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้ทำการปิดแอร์ให้แล้วค่ะ สบายใจได้หายห่วง ')
                elif u'ไฟไหม้' in transcript:
                    print('ตำรวจดับเพลิงกำลังมาครับ')
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้เรียกรถดับเพลิงให้อย่างเร่งด่วนแล้วค่ะ รอสักแป๊ปนะค่ะ ')
                elif u'เปิดทีวี' in transcript:
                    turn_on_tv()
                    print('turning on TV')
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้ทำการเปิดทีวีให้แล้วค่ะ คอยติดตามชมนะค่ะ ')
                elif u'ปิดทีวี' in transcript:
                    turn_off_tv()
                    print('turning off TV')
                    time.sleep(2)
                    text_to_speech(u'อัลโต้ได้ทำการปิดทีวีให้แล้วค่ะ คอยติดตามชมนะค่ะ ')
                elif u'ออก' in transcript:
                    print('Exiting..')
                    break
                else:
                    print("text not matched transcript")
                    print(transcript)

            num_chars_printed = 0
    except:
        print("error in listen_print_loop")

def turn_on_ac():
    # My API
    # PUT https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/9b86fd56-4de8-4b1d-b2de-98c3f5243e27
    print("sending turning on AC command")
    try:
        response = requests.put(
            url="https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/9b86fd56-4de8-4b1d-b2de-98c3f5243e27",
            headers={
                "Authorization": "Bearer 9e450b2e-3acf-4494-8183-c01806684bd2",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps({
                "command": "on"
            })
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

def turn_off_ac():
    # My API
    # PUT https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/9b86fd56-4de8-4b1d-b2de-98c3f5243e27
    print("sending turning off AC command")
    try:
        response = requests.put(
            url="https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/9b86fd56-4de8-4b1d-b2de-98c3f5243e27",
            headers={
                "Authorization": "Bearer 9e450b2e-3acf-4494-8183-c01806684bd2",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps({
                "command": "off"
            })
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

def turn_on_light():
    # front lifx
    # PUT https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/08e75403-af24-4dcf-b1d6-ce253008858a
    print("sending turning on Light command")
    try:
        response = requests.put(
            url="https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/08e75403-af24-4dcf-b1d6-ce253008858a",
            headers={
                "Authorization": "Bearer 9e450b2e-3acf-4494-8183-c01806684bd2",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps({
                "command": "on"
            })
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

def turn_off_light():
    # front lifx
    # PUT https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/08e75403-af24-4dcf-b1d6-ce253008858a
    print("sending turning off Light command")
    try:
        response = requests.put(
            url="https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/08e75403-af24-4dcf-b1d6-ce253008858a",
            headers={
                "Authorization": "Bearer 9e450b2e-3acf-4494-8183-c01806684bd2",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps({
                "command": "off"
            })
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

def turn_on_tv():
    # TV
    # PUT https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/b64f5e3a-d447-48f8-8d86-050de41cec7a
    print("sending turning on TV command")
    try:
        response = requests.put(
            url="https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/b64f5e3a-d447-48f8-8d86-050de41cec7a",
            headers={
                "Cookie": "JSESSIONID=FCCFD41E2393191D2AA158F251B09CDF-n2",
                "Authorization": "Bearer 9e450b2e-3acf-4494-8183-c01806684bd2",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps({
                "command": "on"
            })
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

def turn_off_tv():
    # TV
    # PUT https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/b64f5e3a-d447-48f8-8d86-050de41cec7a
    print("sending turning off TV command")
    try:
        response = requests.put(
            url="https://graph.api.smartthings.com/api/smartapps/installations/a0304624-dc42-4d93-b84f-421ff52167ca/switches/b64f5e3a-d447-48f8-8d86-050de41cec7a",
            headers={
                "Cookie": "JSESSIONID=FCCFD41E2393191D2AA158F251B09CDF-n2",
                "Authorization": "Bearer 9e450b2e-3acf-4494-8183-c01806684bd2",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps({
                "command": "off"
            })
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

def text_to_speech(msg):
    # Request (4)
    # POST http://localhost:8888/test

    try:
        response = requests.get(
            url="http://localhost:8080/websocket",
            # headers={
            #     "Cookie": "_xsrf=2|944ae2e2|73e716843e1dd25abd7365e04aec06e7|1490541278",
            #     "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            # },
            # data={
            #     "message": msg,
            # },
        )
        print('Response HTTP Status Code: {status_code}'.format(
            status_code=response.status_code))
        print('Response HTTP Response Body: {content}'.format(
            content=response.content))
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

def main():
    service = cloud_speech_pb2.SpeechStub(
        make_channel('speech.googleapis.com', 443))

    # For streaming audio from the microphone, there are three threads.
    # First, a thread that collects audio data as it comes in
    with record_audio(RATE, CHUNK) as buffered_audio_data:
        # Second, a thread that sends requests with that data
        requests = request_stream(buffered_audio_data, RATE)
        # Third, a thread that listens for transcription responses
        recognize_stream = service.StreamingRecognize(
            requests, DEADLINE_SECS)

        # Exit things cleanly on interrupt
        signal.signal(signal.SIGINT, lambda *_: recognize_stream.cancel())

        # Now, put the transcription responses to use.
        try:
            listen_print_loop(recognize_stream)
            recognize_stream.cancel()
        except grpc.RpcError as e:
            code = e.code()
            # CANCELLED is caused by the interrupt handler, which is expected.
            if code is not code.CANCELLED:
                raise

    # tornado.options.parse_command_line()
    # app = Application()
    # app.listen(options.port)
    # tornado.ioloop.IOLoop.current().start()

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: chat.proto
# Protobuf Python Version: 6.30.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    30,
    0,
    '',
    'chat.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nchat.proto\x12\x04\x63hat\"!\n\x0eMessageRequest\x12\x0f\n\x07message\x18\x01 \x01(\t\" \n\x0fMessageResponse\x12\r\n\x05reply\x18\x01 \x01(\t2I\n\x0b\x43hatService\x12:\n\x0bSendMessage\x12\x14.chat.MessageRequest\x1a\x15.chat.MessageResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'chat_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_MESSAGEREQUEST']._serialized_start=20
  _globals['_MESSAGEREQUEST']._serialized_end=53
  _globals['_MESSAGERESPONSE']._serialized_start=55
  _globals['_MESSAGERESPONSE']._serialized_end=87
  _globals['_CHATSERVICE']._serialized_start=89
  _globals['_CHATSERVICE']._serialized_end=162
# @@protoc_insertion_point(module_scope)

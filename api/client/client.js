const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const path = require('path');

const PROTO_PATH = path.join(__dirname, 'protos', 'chat.proto');
const packageDef = protoLoader.loadSync(PROTO_PATH, {});
const grpcObj = grpc.loadPackageDefinition(packageDef);
const chatPackage = grpcObj.chat;

const client = new chatPackage.ChatService(
  'localhost:50051',
  grpc.credentials.createInsecure()
);

module.exports = client;
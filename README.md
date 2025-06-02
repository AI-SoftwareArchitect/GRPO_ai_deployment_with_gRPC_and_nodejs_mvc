# AI Chat Application

## Proje Tanımı

Bu proje, bir yapay zekâ sohbet uygulamasının iki ana bileşenini içerir:

1. **AI gRPC Servisi**: Yapay zekâ modelini barındıran ve gRPC protokolü üzerinden hizmet veren bir sunucu.
2. **Node.js MVC Uygulaması**: Kullanıcı arayüzünü sağlayan ve AI servisiyle gRPC üzerinden iletişim kuran bir web uygulaması.

## Proje Yapısı
ai-chat-app/
├── ai-api/ # Yapay zekâ gRPC servisi
│ ├── proto/ # .proto tanımları
│ ├── server.js # gRPC sunucu uygulaması
│ └── package.json
│
├── nodejs-mvc/ # Node.js MVC web uygulaması
│ ├── controllers/
│ ├── views/ # EJS şablonları
│ ├── public/ # CSS ve statik dosyalar
│ ├── routes/
│ ├── app.js
│ └── package.json
│
└── README.md


## Kurulum ve Çalıştırma

### 1. AI gRPC Servisini Başlatma

```bash
cd ai-api
npm install
node server.js
```
2. Node.js MVC Uygulamasını Başlatma
bash
cd nodejs-mvc
npm install
node app.js
Uygulama, varsayılan olarak http://localhost:3000 adresinde çalışacaktır.

Özellikler
Gerçek Zamanlı Sohbet: Kullanıcılar, GPT benzeri bir arayüz üzerinden yapay zekâ ile sohbet edebilir.

gRPC İletişimi: Web uygulaması, AI servisiyle gRPC protokolü üzerinden iletişim kurar.

MVC Mimarisi: Uygulama, Model-View-Controller (MVC) mimarisiyle yapılandırılmıştır.

EJS Şablonları: Dinamik HTML içerikleri için EJS şablon motoru kullanılmıştır.

Kullanılan Teknolojiler
Node.js: Sunucu tarafı JavaScript çalıştırma ortamı.

Express.js: Web uygulaması çatısı.

gRPC: Yüksek performanslı, açık kaynaklı uzak prosedür çağrısı (RPC) sistemi.

EJS: JavaScript ile HTML şablonları oluşturmak için kullanılan şablon motoru.

CSS: Kullanıcı arayüzü stillendirmesi.

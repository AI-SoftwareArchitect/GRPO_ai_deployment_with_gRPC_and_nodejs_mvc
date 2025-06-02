const client = require('../client/client');

exports.renderChat = (req, res) => {
  res.render('chat', { messages: [] });
};

exports.handleChat = (req, res) => {
  const message = req.body.message;

  client.SendMessage({ message }, (err, response) => {
    if (err) return res.status(500).send('Hata: gRPC iletişim başarısız');

    res.render('chat', {
      messages: [
        { role: 'user', text: message },
        { role: 'ai', text: response.reply },
      ],
    });
  });
};

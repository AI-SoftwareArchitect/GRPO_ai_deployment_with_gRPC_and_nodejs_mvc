const express = require('express');
const router = express.Router();
const chatController = require('../controllers/chat.controller');

router.get('/', chatController.renderChat);
router.post('/send', chatController.handleChat);

module.exports = router;
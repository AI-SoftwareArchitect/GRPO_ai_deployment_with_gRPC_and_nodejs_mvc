const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const chatRoutes = require('./routes/chat.routes');

const app = express();
const PORT = 3000;

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.urlencoded({ extended: false }));

app.use('/', chatRoutes);

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));

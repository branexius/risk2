#!/usr/bin/env python
# pylint: disable=C0116
# This program is dedicated to the public domain under the CC0 license.

import logging
import telegram
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

bot = telegram.Bot("SUCK_MY_DICK")
# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, _: CallbackContext) -> None:
    update.message.reply_text('Hi! Use /risk to suck dick')


def risk(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    bot.send_photo(chat_id=chat_id, photo=open('Figure_1.png', 'rb'))



def main() -> None:
    """Run bot."""
    # Create the Updater and pass it your bot's token.

    updater = Updater("SUCK_MY_DICK")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("risk", risk))
    dispatcher.add_handler(CommandHandler("help", start))


    # Start the Bot
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()

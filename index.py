import random
import logging
import copy
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, CallbackContext

from concurrent.futures import ThreadPoolExecutor, as_completed
import setproctitle



# Telegram Bot Token
TOKEN="<TOKEN>"
BOT_NAME="<BOT_NAME>"

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(lineno)s - %(funcName)20s() ] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dictionary to store games by chat ID
games = {}

EMOJI_MAP = {
    None: '⚪',
    'player1': '🔴',
    'player2': '🔵',
    'bot': '🔵'
}

LETTER_EMOJIS = ['🇦', '🇧', '🇨', '🇩', '🇪', '🇫', '🇬']

def choose_column(update: Update, context: CallbackContext, player_turn: int) -> None:
    if update.message:
        current_user = update.message.from_user
    elif update.callback_query:
        current_user = update.callback_query.from_user
    else:
        logger.error("Could not determine user from update")
        return
        
    chat_id = update.effective_message.chat_id

    player_turn = games[chat_id]['current_turn']

#    if player_turn != current_user.id:
#        context.bot.send_message(chat_id, "It's not your turn!")
#        return

    board = games[chat_id]['board']
    available_columns = [i for i, col in enumerate(board[0]) if col is None]

    keyboard = [
        [InlineKeyboardButton(LETTER_EMOJIS[col], callback_data=f'column_{col}') for col in available_columns]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    context.bot.send_message(chat_id, 'Choose a column:', reply_markup=reply_markup)

def display_board(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_message.chat_id
    board = games[chat_id]['board']
    available_columns = [LETTER_EMOJIS[i] for i, col in enumerate(board[0])]

    board_representation = ' '.join(available_columns) + '\n'
    for row in board:
        row_representation = ' '.join([EMOJI_MAP[cell] for cell in row])
        board_representation += row_representation + '\n'

    if games[chat_id]['current_turn'] == 'bot':
        current_player_name = BOT_NAME
    else:
        current_player_name = games[chat_id]['current_turn'].username if games[chat_id]['current_turn'] else "Unknown"
    context.bot.send_message(chat_id, f"{board_representation}\n@{current_player_name}'s turn")

def join(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_message.chat_id
    user = update.message.from_user

    # Check if a game for the chat exists, if not, create one.
    if chat_id not in games:
        games[chat_id] = initialize_game_data()

    # Add player to the game of this chat
    if user not in games[chat_id]['players']:
        if len(games[chat_id]['players']) < 2:
            games[chat_id]['players'].append(user)
            update.message.reply_text(f'{user.first_name} has joined the game!')
            logger.info(f'User {user.first_name} has joined the game in chat {chat_id}')
        else:
            update.message.reply_text('The current game in this chat is already full.')
            logger.warning(f'User {user.first_name} tried to join a full game in chat {chat_id}')
    else:
        update.message.reply_text('You are already in the game of this chat.')
        logger.info(f'User {user.first_name} tried to join the game again in chat {chat_id}')

def start_game(update: Update, context: CallbackContext) -> None:
    chat_id = update.effective_message.chat_id

    if chat_id not in games:
        update.effective_message.reply_text('No game has been initiated in this chat. Please join first.')
        logger.warning(f'Attempt to start a game that has not been initiated in chat {chat_id}')
        return

    if len(games[chat_id]['players']) == 1:
        update.message.reply_text('You are playing against the bot!')
        logger.info(f'User is playing against the bot in chat {chat_id}')
        games[chat_id]['current_turn'] = 'bot'
        display_board(update, context)
        bot_move(update, context) 
    elif len(games[chat_id]['players']) == 2:
        update.message.reply_text('The game starts now!')
        games[chat_id]['current_turn'] = random.choice(games[chat_id]["players"])
        display_board(update, context)
        player_turn = games[chat_id]['current_turn']
        choose_column(update, context, player_turn)
        logger.info(f'Game started in chat {chat_id}')
    else:
        update.message.reply_text('We need at least two players to start the game.')
        logger.warning(f'Attempt to start a game with insufficient players in chat {chat_id}')
        return

def place_stone(board, col, player):
    """Place the player's stone in the given column if possible."""
    for row in reversed(board):
        if row[col] is None:
            row[col] = player
            return True
    return False

def is_draw(board):
    """Check if the board is full, resulting in a draw."""
    return all(cell is not None for row in board for cell in row)

def column_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    chat_id = query.message.chat_id
    user = query.from_user

    if games[chat_id]['current_turn'] != user:
        query.answer("It's not your turn!")
        return

    col = int(query.data.split('_')[1])

    current_player = 'player1' if games[chat_id]['players'][0] == user else 'player2'
    if not place_stone(games[chat_id]['board'], col, current_player):
        query.answer("This column is full. Choose another one.")
        return

    if check_winner(games[chat_id]['board'], current_player):
        display_board(update, context)
        context.bot.send_message(chat_id, f'@{user.username} wins!')
        games[chat_id] = initialize_game_data()
        return

    if is_draw(games[chat_id]['board']):
        display_board(update, context)
        context.bot.send_message(chat_id, 'It\'s a draw!')
        games[chat_id] = initialize_game_data()
        return

    # Switch to the other player or the bot if the game is against the bot
    if len(games[chat_id]['players']) == 1:
        games[chat_id]['current_turn'] = 'bot'
        display_board(update, context)
        bot_move(update, context)
    else:
        games[chat_id]['current_turn'] = games[chat_id]['players'][0] if games[chat_id]['current_turn'] == games[chat_id]['players'][1] else games[chat_id]['players'][1]
        display_board(update, context)
        choose_column(update, context, games[chat_id]['current_turn'])

    query.answer()

    # Change the current_turn and continue the game...

def bot_move(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    chat_id = update.effective_message.chat_id

    logger.debug(f"Starting bot_move for chat_id: {chat_id}")

    board = games[chat_id]['board']
    logger.debug(f"Current board: {board}")

    # Use the best_move function to get the best column for the bot's move
    chosen_column = best_move(board)
    logger.debug(f"Chosen column after best_move: {chosen_column}")

    # Now, make the actual move on the board
    if chosen_column != -1 and is_valid_move(board, chosen_column):
        make_move(board, chosen_column, 'bot')
    else:
        # Handle the case where the Minimax algorithm didn't return a valid column
        # This should not normally happen if everything is implemented correctly
        available_columns = [i for i, col in enumerate(board[0]) if col is None]
        chosen_column = random.choice(available_columns)  # Choose a column at random as a fallback
        logger.debug(f"Choosing a random column as fallback: {chosen_column}")

    logger.info(f'Bot placed its move in column {chosen_column}')

    if check_winner(games[chat_id]['board'], 'bot'):
        display_board(update, context)
        context.bot.send_message(chat_id, f'@{BOT_NAME} wins!')
        games[chat_id] = initialize_game_data()
        return

    # Now display the board and ask the user to choose a column
    player_turn = games[chat_id]['players'][0]
    games[chat_id]['current_turn'] = games[chat_id]['players'][0]
    display_board(update, context)
    choose_column(update, context, player_turn)

def main():
    setproctitle.setproctitle('connectfour')

    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('join', join))
    dp.add_handler(CommandHandler('start', start_game))
    dp.add_handler(CallbackQueryHandler(column_callback, pattern='^column_\d'))

    updater.start_polling()
    logger.info('Bot has started.')
    updater.idle()
    logger.info('Bot has shut down.')

def initialize_game_data():
    return {
            'players': [],
            'board': [[None for _ in range(7)] for _ in range(6)],
            'current_turn': None,
        }

# MINIMAX Algorithmus

def is_valid_move(board: list, column: int) -> bool:
    """
    Check if a move is valid.
    Returns True if the move is valid, otherwise False.
    """
    valid = board[0][column] is None
    logger.debug(f"Checking validity of move for column {column}. Is valid: {valid}. Board's first row value at column {column}: {board[0][column]}")
    return valid

def has_valid_moves(board):
    """
    Überprüft, ob es auf dem Spielbrett noch gültige Züge gibt.
    
    :param board: Das aktuelle Spielbrett.
    :return: True, wenn mindestens ein gültiger Zug vorhanden ist, sonst False.
    """
    
    for col in range(len(board[0])):  # Gehe durch jede Spalte
        if board[0][col] is None:  # Wenn das oberste Feld der Spalte leer ist (None), gibt es einen gültigen Zug
            return True
    return False  # Keine gültigen Züge gefunden

def evaluate_board(board):
    """
    Evaluate the given board state and return a heuristic value.
    """
    # The weights can be adjusted based on the importance of each pattern
    SCORES = {
        'bot_4': 100,
        'bot_3': 5,
        'bot_2': 2,
        'player_4': -100,
        'player_3': -4,
        'player_2': -1
    }
    
    # This function checks for sequences of same cells
    def count_sequences(board, player, length):
        count = 0
        
        # Horizontal check
        for row in board:
            for i in range(len(row) - length + 1):
                if all([cell == player for cell in row[i:i+length]]):
                    count += 1

        # Vertical check
        for j in range(len(board[0])):
            for i in range(len(board) - length + 1):
                if all([board[k][j] == player for k in range(i, i+length)]):
                    count += 1

        # Diagonal checks
        for i in range(len(board) - length + 1):
            for j in range(len(board[0]) - length + 1):
                if all([board[i+k][j+k] == player for k in range(length)]):
                    count += 1

        for i in range(len(board) - length + 1):
            for j in range(length - 1, len(board[0])):
                if all([board[i+k][j-k] == player for k in range(length)]):
                    count += 1
        
        return count

    # Evaluate sequences for both bot and player
    total_score = 0
    for player in ['bot', 'player']:
        for length in [2, 3, 4]:
            total_score += SCORES[f"{player}_{length}"] * count_sequences(board, player, length)
    
    return total_score

def minimax(board, depth, isMaximizingPlayer, alpha=float('-inf'), beta=float('inf')):
    logging.debug(f"Entered minimax with depth: {depth} and isMaximizingPlayer: {isMaximizingPlayer}")

    # Check if the game is over
    winner = check_winner(board, 'bot')
    if winner == 'bot':
        logging.debug("Winner is 'bot'")
        return 10 - depth
    elif winner == 'player':
        logging.debug("Winner is 'player'")
        return depth - 10
    elif winner == 'draw':
        logging.debug("It's a draw")
        return 0

    # If depth is 0 or no more moves, return a heuristic value
    if depth == 0 or not has_valid_moves(board):
        heuristic_value = evaluate_board(board)
        logging.debug(f"Depth is 0 or no more valid moves. Heuristic value: {heuristic_value}")
        return heuristic_value

    # Move ordering (check middle columns first for Connect Four for instance)
    ordered_moves = [3, 2, 4, 1, 5, 0, 6]  # Can be adjusted based on the board size and game logic

    # If this is the maximizing player (the bot)
    if isMaximizingPlayer:
        maxEval = float('-inf')
        for col in ordered_moves:
            if is_valid_move(board, col):
                temp_board = copy.deepcopy(board)
                make_move(temp_board, col, 'bot')
                eval = minimax(temp_board, depth-1, False, alpha, beta)
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:  # Pruning
                    break
        logging.debug(f"Maximizing player's evaluation: {maxEval}")
        return maxEval
    else:  # If this is the minimizing player (the opponent)
        minEval = float('inf')
        for col in ordered_moves:
            if is_valid_move(board, col):
                temp_board = copy.deepcopy(board)
                make_move(temp_board, col, 'player')
                eval = minimax(temp_board, depth-1, True, alpha, beta)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:  # Pruning
                    break
        logging.debug(f"Minimizing player's evaluation: {minEval}")
        return minEval

def evaluate_move(board, col):
    temp_board = copy.deepcopy(board)
    make_move(temp_board, col, 'bot')
    eval = minimax(temp_board, 5, False)  # Adjust the depth as needed
    return col, eval

def best_move(board):
    maxVal = float('-inf')
    bestColumn = -1
    
    with ThreadPoolExecutor() as executor:
        # Start all the tasks and get a generator that yields Futures
        futures = {executor.submit(evaluate_move, board, col): col for col in range(len(board[0])) if is_valid_move(board, col)}
        
        for future in as_completed(futures):
            col, eval = future.result()
            if eval > maxVal:
                maxVal = eval
                bestColumn = col

    return bestColumn

def make_move(board: list, column: int, player: str) -> bool:
    """
    Make a move on the board. Returns True if the move was successful, otherwise False.
    """
    for row in reversed(board):
        if row[column] is None:
            row[column] = player
            return True
    return False

def check_winner(board, player):
    """Check if the given player has won."""
    rows, cols = len(board), len(board[0])

    # Check horizontal
    for row in range(rows):
        for col in range(cols - 3):  # -3 to ensure there's space for 4 in a row
            if all(board[row][col + i] == player for i in range(4)):
                return True

    # Check vertical
    for row in range(rows - 3):  # -3 to ensure there's space for 4 in a row
        for col in range(cols):
            if all(board[row + i][col] == player for i in range(4)):
                return True

    # Check diagonal from top-left to bottom-right
    for row in range(rows - 3):  # -3 to ensure there's space for 4 in a diagonal
        for col in range(cols - 3):  # -3 to ensure there's space for 4 in a diagonal
            if all(board[row + i][col + i] == player for i in range(4)):
                return True

    # Check diagonal from bottom-left to top-right
    for row in range(3, rows):  # Start from 3 to ensure there's space for 4 in a diagonal
        for col in range(cols - 3):  # -3 to ensure there's space for 4 in a diagonal
            if all(board[row - i][col + i] == player for i in range(4)):
                return True

    return False

if __name__ == '__main__':
    main()

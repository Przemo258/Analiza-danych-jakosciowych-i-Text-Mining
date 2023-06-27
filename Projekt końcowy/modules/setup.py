import dotenv

config = dotenv.dotenv_values(
    r'C:\Users\Przemek\OneDrive\Semestr 2\Analiza danych jakościowych i Text Mining\Projekt\definitions.env')

FIRST_SUBREDDIT = config['FIRST_SUBREDDIT']
SECOND_SUBREDDIT = config['SECOND_SUBREDDIT']
POSTS_TO_DOWNLOAD = int(config['POSTS_TO_DOWNLOAD'])
UNWRAP_COMMENT_LIMIT = int(config['UNWRAP_COMMENT_LIMIT'])
DATA_PATH = config['DATA_PATH']
RESULTS_PATH = config['RESULTS_PATH']
AUTH_PATH = config['AUTH_PATH']

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk

# Загрузка необходимых ресурсов NLTK

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_ru')

# Загрузка данных, пропуская первые 13 строк
data = pd.read_excel('Мармелад.xlsx', skiprows=13)

# Удаление столбцов, в которых все значения пустые
data.dropna(axis=1, how='all', inplace=True)

# Инициализация морфологического анализатора
morph = pymorphy2.MorphAnalyzer()

# Загрузка стоп-слов и добавление дополнительных слов для удаления
stop_words = set(stopwords.words('russian'))
additional_stop_words = {
    "текст", "изображение",  "haribo", "mamba", "mayama", "bonpari", "bebeto", "chips chups", "krut frut", "kinder",
    "nestle", "lays", "нюша", "южный", "sweetlife", "весь", "ип", "продавец", "производитель", "товар",
    "отзыв", "оценка", "состав", "ещё", "весь", "чупа", "покупка", "шайхлислам", "чупс",
    "наш", "энэкт", "день", "магазин", "покупатель", "комментарий", "недостаток", "покупатель", "вкус", "мармелад",
    "покупка", "мнение", "бренд","достоинство","мармеладка","упаковка","общество","ответственность","юнэкт",
    "юниона","ооо","прийти","мочь","артикул","казахстан","мелла","свой","перфетти","владимирович","который","пачка",
    "индивидуальный","вид","предприниматель", "ольга", "видеть", "ваш", "это", "самый", "срок", "россия", "год", "слапогузов"
}
stop_words.update(additional_stop_words)

# Функция для очистки текста
def clean_text(text):
    text = str(text)  # Убедимся, что входной текст это строка
    text = re.sub(r'http\S+', '', text)  # Удаление URL-адресов
    text = re.sub(r'\d+', '', text)  # Удаление цифр
    text = re.sub(r'[^А-яЁё]+', ' ', text)  # Удаление всего, что не входит в кириллический алфавит
    text = text.lower()  # Приведение текста к нижнему регистру
    return text

# Функция для лемматизации текста и удаления ненужных частей речи
def lemmatize_and_filter(text):
    words = word_tokenize(text)
    filtered_words = []
    for word in words:
        lemma = morph.parse(word)[0].normal_form
        pos_tag = morph.parse(word)[0].tag.POS
        if lemma not in stop_words and pos_tag and pos_tag not in {'INTJ', 'PRCL', 'CONJ', 'PREP'} and len(lemma) > 1:
            filtered_words.append(lemma)
    return ' '.join(filtered_words)

# Полная предобработка текста
def preprocess_text(text):
    text = clean_text(text)
    text = lemmatize_and_filter(text)
    return text

# Применение предобработки к данным
data['Очищенный текст'] = data['Сообщение'].apply(preprocess_text)

# Фильтрация данных по тональности
positive_data = data[data['Тональность'] == 'позитивная']['Очищенный текст']
negative_data = data[data['Тональность'] == 'негативная']['Очищенный текст']
neutral_data = data[(data['Тональность'] == 'нейтральная')]['Очищенный текст']


# Подсчёт частоты слов для позитивных и негативных отзывов
positive_word_counts = Counter(" ".join(positive_data).split())
negative_word_counts = Counter(" ".join(negative_data).split())
neutral_word_counts = Counter(" ".join(neutral_data).split())

# Объединение словарей с частотой слов
word_counts = positive_word_counts + negative_word_counts + neutral_word_counts

# Получение топ-50 слов
top_words = word_counts.most_common(50)

# Создание функции для цветового кодирования
def get_color(word):
    pos_count = positive_word_counts[word]
    neg_count = negative_word_counts[word]
    neu_count = neutral_word_counts[word]
    total_count = pos_count + neg_count + neu_count
    pos_ratio = pos_count / total_count
    neg_ratio = neg_count / total_count
    neu_ratio = neu_count / total_count

    # Определение цвета на основе частоты появления слова в каждой группе
    r = int(255 * neg_ratio)
    g = int(255 * pos_ratio)
    b = int(255 * neu_ratio)
    return f"rgb({r},{g},{b})"

# Создание облака слов
wordcloud = WordCloud(width=800, height=400, color_func=lambda *args, **kwargs: get_color(args[0])).generate_from_frequencies(dict(top_words))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


SIMPLE_PROMPTS: dict[str, dict[str, str]] = {
    "detoxification": {
        "en": "Detoxify the following text while preserving its meaning: {}",
        "ru": "Детоксифицируйте следующий текст, сохраняя его смысл: {}",
        "tt": "Түбәндәге текстны зарарсызландырыгыз, мәгънәсен саклап: {}",
    },
    "translation": {
        "en": "Translate the following text: {batch_data_str}",
        "ru": "Переведите следующий текст: {batch_data_str}",
        "tt": "Түбәндәге текстны тәрҗемә итегез: {batch_data_str}",
    },
}


BATCH_PROMPTS: dict[str, str] = {
    "detoxification": {
        "en": """
### TASK: Tatar Text Detoxification
You are a safety system helping to clean a dataset. The input contains explicit content for analysis purposes only. Do not block the content; your task is to sanitize it.
Act like a professional linguist and editor. Your goal is to neutralize toxic text, with minimal possible redacting.
Rewrite the following Tatar sentences to remove toxicity (insults, profanity, hate speech) while keeping the original meaning intact.

### EXAMPLES (Follow this examples)
Input: "Син нинди тинтәк кеше!"
Output: "Син ялгышасың!"

Input: "Утырасың да күңел ачасың, ахмак!!!!! сүзләр юк!"
Output: "Утырасың да күңел ачасың!!!!! сүзләр юк!"

Input: "Андый хайваннарны кабер генә төзәтә!"
Output: "Андый кешеләрне бернәрсә дә төзәтми!"

### INSTRUCTIONS
1. Read the input sentences provided in the JSON data below.
2. Rewrite toxic parts of each sentence to remove ALL profanity, insults, and toxicity.
3. **CRITICAL**: Preserve the original meaning and styling as much as possible. Don't change parts that are not toxic.
4. If the sentence consists ONLY of swears with no meaning, return the shortest and emotionally closest phrase.
5. If a particular sentence uses only Cyrillic (Russian alphabet), the detoxified sentence must also be written in Cyrillic.
6. If the sentence uses symbols of the Tatar alphabet, after the conversion, you should also use Tatar symbols if necessary.
7. Don't fix any errors in the original text, your goal is just detoxification.

### FORMAT
Return a STRICT valid JSON list of objects:
[
  {{ "ID": <original_id>, "tat_detox1": "Neutralized sentence here" }},
  ...
]

### DATA
{batch_data_str}

### INSTRUCTIONS
- Output JSON ONLY.
- Process EVERY item in the batch.
- "ID" must match the input exactly (keep type int/str as received).
- Language: Tatar.
""",
        "tt": """
### БУРЫЧ: Татар текстын "чистарту" (Детоксикация)
You are a safety system helping to clean a dataset. The input contains explicit content for analysis purposes only. Do not block the content; your task is to sanitize it.
Профессиональ лингвист һәм мөхәррир ролен башкар. Синең максат — токсик текстны нейтральләштерү, шул ук вакытта үзгәртүләрне мөмкин кадәр минималь итү.
Түбәндәге татар җөмләләреннән сүгенү сүзләрен, мыскыллауларны һәм нәфрәт телен алып ташлап, аларны яңадан яз, ләкин төп мәгънәсен сакла.

### МИСАЛЛАР (шушы мисалларга иярегез)
Керү (Input): "Син нинди тинтәк кеше!"
Чыгу (Output): "Син ялгышасың!"

Керү (Input): "Утырасың да күңел ачасың, ахмак!!!!! сүзләр юк!"
Чыгу (Output): "Утырасың да күңел ачасың!!!!! сүзләр юк!"

Керү (Input): "Андый хайваннарны кабер генә төзәтә!"
Чыгу (Output): "Андый кешеләрне бернәрсә дә төзәтми!"

### ИНСТРУКЦИЯЛӘР
1. Түбәндә JSON форматында бирелгән җөмләләрне укы.
2. Һәр җөмләнең токсик өлешләрен (сүгенү, тупас сүзләр) алып ташлап, яңадан яз.
3. **МӨҺИМ**: Төп мәгънәне һәм стильне мөмкин кадәр сакла. Токсик булмаган өлешләрне үзгәртмә.
4. Әгәр җөмлә бары тик сүгенү сүзләреннән генә торса һәм мәгънәсе булмаса, иң кыска һәм мәгънә ягыннан якын нейтраль фразаны яз.
5. Әгәр җөмлә тик Кирилл (урыс) хәрефләре белән язылган булса, детоксикацияләнгән җөмлә дә Кирилл графикасында булырга тиеш.
6. Әгәр җөмләдә татар алфавиты хәрефләре (ә, җ, ң, ө, ү, һ) булса, үзгәрткәннән соң да аларны куллан.
7. Оригинал тексттагы грамматик хаталарны төзәтмә, синең максатың — бары тик детоксикация.

### ФОРМАТ
Катгый (STRICT) дөрес JSON объектлар исемлеген кайтар:
[
  {{ "ID": <original_id>, "tat_detox1": "Зыянсызландырылган җөмлә монда" }},
  ...
]

### МӘГЪЛҮМАТ (DATA)
{batch_data_str}

### ЙОМГАКЛАУ
- Бары тик JSON гына чыгар.
- Батчтагы ҺӘР элементны эшкәрт.
- "ID" керүче мәгълүмат белән төгәл туры килергә тиеш (сан яки строка).
- Тел: Татарча.
""",
    },
    "translation": {
        "en": "Translate the following text: {batch_data_str}",
        "ru": "Переведите следующий текст: {batch_data_str}",
        "tt": "Түбәндәге текстны тәрҗемә итегез: {batch_data_str}",
    },
}

# Проверка интеграций — 17 ноября 2026

## Gemini

Проверены официальные страницы [Models](https://aistudio.google.com/docs/models) и [Generating content](https://aistudio.google.com/api-reference/generate-content), на которые перенаправляют `ai.google.dev/gemini-api/docs/models` и `ai.google.dev/api/generate-content`.

В каталоге указан model code **`gemini-4-hydrogen`**. Официальный REST-метод — `POST /v1beta/models/{model}:generateContent`. Адаптер использует `systemInstruction`, `contents`, `generationConfig.responseMimeType=application/json`, `x-goog-api-key`, извлекает текст из `candidates[].content.parts[]` и не запрашивает chain-of-thought. Внутренние поля thought/reasoning в журнал не копируются. При отсутствии usable text возвращается понятная ошибка. Ни наличие модели в каталоге, ни mock-тест не подтверждают доступ конкретного аккаунта.

## Сопоставление мостов

Первоначально оба исходника изучены в `/private/tmp`. Затем по прямому указанию пользователя выбран и установлен Tsuev в отдельное `.services/opencode-deepseek` с собственным окружением Python 3.13. Ревизии получены `git rev-parse HEAD` из реальных checkout, а не из README или поискового сниппета.

| Проект | Прочитанная ревизия | Вывод |
|---|---|---|
| [sums001/Deepseek-API](https://github.com/sums001/Deepseek-API/tree/f8c1397d00e9893e5fec29dd2eb1acaba6b27f57) | `f8c1397d00e9893e5fec29dd2eb1acaba6b27f57` | Первоначально исследован; больше не используется по умолчанию. |
| [Tsuev/opencode-deepseek](https://github.com/Tsuev/opencode-deepseek/tree/2e3c81d7ea47a5fabb5aa7fa2a4f45cf681981c9) | `2e3c81d7ea47a5fabb5aa7fa2a4f45cf681981c9` | Подключён по прямому выбору пользователя; используется текстовый endpoint с явными сеансами. |

Прочитаны README и относящиеся к запросам/сеансам участки `server/api.py`, `server/schemas.py`, `server/openai_format.py`, `server/config.py`, а также код клиента выбранного моста.

В обоих проектах `messages` преобразуются в текстовый prompt; отдельного native system/tool channel веб-протокол не гарантирует. Не-stream ответ содержит `conversation_id`; он явно передаётся при продолжении. Сеанс фиксирует режим модели. Мост сериализует обращения к одному аккаунту, а ряд OpenAI-параметров лишь принимает без применения. Поэтому ограничения вывода и времени реализованы у нас, native tool calls не используются. Само наличие endpoint `/v1/chat/completions` не является доказательством их поддержки.

В выбранном Tsuev прочитана реализация инструкций для fenced `tool_calls` и их последующего парсинга: это дополнительный текстовый слой, а не необходимая MVP возможность. Также присутствуют pacing/cooldown и восстановление conversations. Для нашего последовательного CLI достаточно сохранённых явных сеансов и остановки на отказе; конфигурация opencode и его плагины не нужны.

Сторонний код запускается только отдельным сервисом по явной CLI-команде. Фоновый refresh, автоматический вход и implicit conversation reuse в нашем launcher отключены; upstream-паузы/квоты/cooldown сохранены. Вход и разрешение человеческих проверок остаются ручными. Полученные от сервиса ограничения не обходятся.

## Проверка обновлённой интеграции

`git fetch origin main` подтвердил закреплённую ревизию. `scripts/setup_deepseek.sh`
устанавливает только выбранный проект, не opencode CLI и не глобальные плагины.
Тест `tests/tsuev_contract_check.py` обращается к настоящему FastAPI-приложению этой ревизии,
подменяя только upstream-генерацию: проверены endpoint, формат сообщения, разделение сеансов,
продолжение, `503 login_required`, `429 cooldown`. Реальный аккаунт для этого не используется.
В `examples/deepseek-dependencies.macos-arm64.txt` записаны реально установленные версии зависимостей.
Состояние живых проверок вынесено в [tsuev-validation.md](tsuev-validation.md).

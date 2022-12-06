# Лабораторная работа 2

## Задание

**Цель работы**:

Получить навыки разработки CI/CD pipeline для ML моделей с достижением метрик моделей и качества.

По имеющимся данным: https://files.grouplens.org/datasets/movielens/ml-20mx16x32.tar построить content-based рекомендации по образовательным курсам.

Запрещено использовать библиотеки pandas, sklearn и аналогичные.

Рекомендации по выполнению работы:
Для подбора рекомендаций следует использовать меру TFIDF, а в качестве метрики для ранжирования — косинус угла между TFIDF-векторами.

**Дополнительное задание**: построить полный CI/CD цикл модели с использованием Docker/Jenkins/подходов тестирования

**Ход работы**:

- [x] Создать репозитории модели на GitHub, регулярно проводить commit + push в ветку разработки, важна история коммитов;
- [x] Провести подготовку данных для набора данных;
- [x] Разработать ML модель;
- [ ] Покрыть код тестами;
- [x] Использовать Docker для создания docker image.
- [ ] Наполнить дистрибутив конфигурационными файлами:
    - - [ ] config.ini: гиперпараметры модели;
    - - [ ] Dockerfile и docker-compose.yml: конфигурация создания контейнера и образа модели;
    - - [ ] requirements.txt: используемые зависимости (библиотеки) и их версии;
- [ ] Создать CI pipeline (Jenkins, Team City, Circle CI и др.) для сборки docker image и отправки его на DockerHub, сборка должна автоматически стартовать по pull request в основную ветку репозитория модели;
- [ ] Создать CD pipeline для запуска контейнера и проведения функционального тестирования по сценарию, запуск должен стартовать по требованию или расписанию или как вызов с последнего этапа CI pipeline;
- [ ] Результаты функционального тестирования и скрипты конфигурации CI/CD pipeline приложить к отчёту.

**Результаты работы**:

1. Отчёт о проделанной работе;
2. Ссылка на репозиторий GitHub;
3. Ссылка на docker image в DockerHub;
4. Актуальный дистрибутив модели в zip архиве.

Обязательно обернуть модель в контейнер (этап CI) и запустить тесты внутри контейнера (этап CD).

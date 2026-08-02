# Аналітичний асистент для райд-хейлінгу

Тестове завдання Data Scientist (Uklon).

**Демо:** https://ride-hailing-analytics.fly.dev/

Синтетичні дані про Київ (60 днів): ~737 тис. розрахунків ціни з воронкою
«розрахунок → замовлення → пошук водія / churn», плюс стан зон що 15 хвилин.
Асистент відповідає природною мовою, пише SQL і будує графіки.

## Що всередині (коротко)

| Частина | Суть |
|---|---|
| Датасет | PostgreSQL: `rides`, `zone_state`, зони Києва, комендантська, тривоги |
| Воронка | calculated → placed → accepted **або** churned_to_competitor (~2–5 хв пошуку) |
| Асистент | LangChain/LangGraph + Claude (OpenRouter), інструменти аналітики + `run_sql` |
| Акцент DS | Парадокс Сімпсона: surge vs acceptance при контролі ETA |

## Швидкий старт (локально)

```bash
make install && make db-up && make reset && make serve
```

Відкрити http://localhost:8000

## Деталі

Повний research (Uklon pricing, що взяли / чим пожертвували), схема даних,
валідація, скріни, деплой і обмеження:

→ **[docs/DETAILED.md](docs/DETAILED.md)**

## Стабільність демо

- Fly: `auto_stop_machines = off`, `min_machines_running = 2`
- Дані вже залиті в керований Postgres; апка не перегенерує їх при старті
- Щоб демо не «падало» тиждень: не робіть `fly deploy` / reload БД без потреби;
  перевіряйте https://ride-hailing-analytics.fly.dev/api/health → `"status":"ok"`

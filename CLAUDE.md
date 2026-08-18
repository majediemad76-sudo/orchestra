# Multi-Model Orchestrator

یک ارکستراتور چهارنقشی: Manager / Worker / Critic / Controller.
هدف: هیچ مدلی کنترل حلقه را در دست نگیرد.

## معماری چهارنقشی

| نقش | مدل | فایل | کارش |
|---|---|---|---|
| Manager | `grok-4.6` (xAI API) | `roles/manager.py`, `providers/xai.py` | تجزیه‌ی تسک، نوشتن `worker_prompt` خودبسنده، تعریف معیارهای پذیرش |
| Worker (متن) | `claude-sonnet-5` (Anthropic API) | `roles/worker.py`, `providers/anthropic.py` | اجرای کار متنی |
| Worker (کد) | Claude Code headless (subprocess) | `providers/claude_code.py` | اجرای کار روی فایل‌سیستم |
| Critic | `gemini-3.1-flash-lite` (Google API) | `roles/critic.py`, `providers/google.py` | نمره‌دهی خروجی Worker در برابر معیارها |
| Controller | کد پایتون معمولی | `controller.py` | ماشین حالت، سقف تکرار، بودجه، ارجاع به کاربر، لاگ |

**چرا سه فروشنده‌ی متفاوت:** Critic باید از فروشنده‌ای غیر از Worker باشد. مدلی که خودش
خروجی را نوشته بدترین داور آن است و مدل هم‌خانواده‌اش هم سوگیری مشابه دارد. این قید معماری
است، نه سلیقه — هنگام تغییر مدل‌ها آن را نشکن.

## قوانین ثابت پروژه

1. **Controller همیشه کد است، هرگز مدل.** هیچ تصمیم قطعی — تعداد retry، زمان پرسیدن از
   کاربر، سقف هزینه — به هیچ مدلی سپرده نمی‌شود. مدل‌ها فقط ورودی تصمیم را تولید می‌کنند.
2. **منطق حلقه در پایتون می‌ماند**، نه در n8n و نه هیچ ابزار گره‌ای دیگر. اگر روزی UI لازم شد،
   UI روی همین `run_task()` سوار شود.
3. **هیچ کلید API هرگز در کد hardcode نمی‌شود.** فقط از `.env` خوانده می‌شود
   (`os.environ` داخل `providers/*.py`). `.env.example` الگو است و `.env` در `.gitignore`.
4. **سه‌تا و فقط سه‌تا تریگر برای ارجاع به کاربر** (پیاده‌سازی در `controller.py`):
   - `two_rejections` — دو رد پشت‌سرهم از Critic
   - `manager_needs_input` — Manager صریحاً `needs_user_input: true` می‌دهد
   - `budget_exceeded` — عبور از سقف دلاری
   هر ارجاع = یک سؤال با **۲ تا ۴ گزینه‌ی مشخص**، نه سؤال باز.
   مواردی که تریگر **نیستند** و داخل حلقه حل می‌شوند: `verdict = escalate` از Critic
   (به‌عنوان یک رد شمرده می‌شود و به Manager برمی‌گردد)، تایم‌اوت Claude Code (مثل یک رد)،
   و رسیدن به سقف دورها (اجرا با گزارش تمام می‌شود).

## خروجی ساخت‌یافته — سطح API، نه پرامپت

جمله‌ی «فقط JSON بده» در پرامپت‌ها فقط پشتیبان است. تکیه‌گاه اصلی `providers/schema_utils.py`
است که اسکیمای Pydantic را به سه دیالکت تبدیل می‌کند (و `$ref`ها را inline می‌کند):

- **Anthropic** — اجبار به فراخوانی ابزار: `tools=[{... "input_schema": schema}]` به‌همراه
  `tool_choice={"type":"tool","name":"emit_result"}`. Claude پارامتر `response_format` ندارد.
- **xAI** — `response_format: {"type":"json_schema","json_schema":{...,"strict":true}}`
  (همان الگوی OpenAI، چون endpoint سازگار با OpenAI است). در حالت strict هر object باید
  `additionalProperties: false` داشته باشد و همه‌ی propertyها در `required` بیایند؛ فیلدهای
  اختیاری به‌جای حذف، nullable می‌شوند.
- **Gemini** — `generationConfig.responseMimeType = "application/json"` و
  `generationConfig.responseSchema` بومی: نوع‌ها با حروف بزرگ
  (`STRING`/`OBJECT`/`ARRAY`/`INTEGER`/`BOOLEAN`) و Optional به `nullable: true`.

## Retry و بودجه

- `providers/retry_utils.py`: روی ۴۲۹، ۵xx و خطای شبکه حداکثر ۳ تلاش با backoff نمایی.
  روی ۴۰۰/۴۰۱ **retry نمی‌کنیم** — کلید اشتباه با صبر کردن درست نمی‌شود.
- `budget.py`: قیمت‌ها (دلار به‌ازای هر یک میلیون توکن، آگوست ۲۰۲۶) —
  `claude-sonnet-5` ۲٫۰۰/۱۰٫۰۰ · `grok-4.6` ۲٫۰۰/۶٫۰۰ · `gemini-3.1-flash-lite` ۰٫۲۵/۱٫۵۰.
  Worker کدی هزینه‌ی دلاری خودش را از خروجی JSON کلاد کد می‌دهد (`charge_usd`).

## Claude Code headless

`subprocess.run([... "--output-format","json","--max-turns","15"], timeout=600)`.
`TimeoutExpired` هرگز نباید برنامه را crash کند؛ به یک رد از Critic ترجمه می‌شود
(`roles/critic.failed_worker_verdict`) تا حلقه بتواند اصلاح و دوباره تلاش کند.

## لاگ

هر اجرا یک فایل `runs/<run_id>.jsonl` می‌سازد. رویدادها: `run_start`، `manager_plan`،
`worker_output`، `critic_verdict`، `escalation`، `escalation_answer`، `budget_raised`،
`run_end` — هرکدام با هزینه‌ی همان مرحله.

## دستورها

```bash
./.venv/bin/python scripts/self_check.py          # بدون کلید API اجرا می‌شود
./.venv/bin/python controller.py "هدف" --budget 0.50 --max-rounds 4
```

`scripts/self_check.py` قبل از هر commit باید سبز باشد؛ بدون کلید و بدون شبکه اجرا می‌شود و
تبدیل‌کننده‌های اسکیما، حساب بودجه و هر سه تریگر ارجاع را با نقش‌های ساختگی تست می‌کند.

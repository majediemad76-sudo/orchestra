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
2. **منطق حلقه در پایتون می‌ماند**، نه در n8n و نه هیچ ابزار گره‌ای دیگر. هر UI روی همین
   `run_task()` سوار می‌شود و فقط نقش Observer دارد — بخش «رابط Streamlit» پایین.
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

## قرارداد Callback کنترلر

`run_task()` سه قلاب اختیاری می‌گیرد. هر سه **افزودنی**‌اند: هیچ‌کدام نمی‌توانند تصمیمی را عوض
کنند، و اگر هیچ‌کدام داده نشود رفتار دقیقاً همان اجرای headless در ترمینال است.

```python
run_task(task, cwd=None, ask=ask_on_console, run_id=None,
         on_progress=None,      # (event: str, data: dict) -> None
         on_escalation=None,    # (Question) -> str   لیبل گزینه‌ی انتخابی
         stop_flag=None)        # threading.Event
```

- **`on_progress`** بعد از هر مرحله صدا زده می‌شود: `round_start`، `manager_plan`،
  `worker_output`، `critic_verdict`، `run_end`. فقط مشاهده است؛ مقدار بازگشتی نادیده گرفته
  می‌شود و استثنای آن کشنده نیست (یک observer معیوب نباید اجرایی را که پولش داده شده از بین
  ببرد). **لاگ JSONL همچنان سند اصلی است**، نه این قلاب.
- **`on_escalation`** جای پرسش ترمینال را در هر سه تریگر می‌گیرد و **لیبل** گزینه را برمی‌گرداند
  (نه ایندکس — چون چیزی که یک UI در دست دارد متن است). پاسخ ناشناخته یا خالی = «بی‌پاسخ» =
  توقف؛ عمداً به هیچ گزینه‌ی پیش‌فرضی تبدیل نمی‌شود.
- **`stop_flag`** در ابتدای هر دور و **قبل از شمردن دور** خوانده می‌شود، پس اجرای متوقف‌شده بابت
  کاری که کسی نمی‌خواند شارژ نمی‌شود. وضعیت نهایی `stopped_by_flag` است و summary کامل با
  دورها و هزینه‌ی تا آن لحظه برمی‌گردد.

## رابط Streamlit — فقط Observer

`app.py` هیچ منطق ارکستراسیونی ندارد و **نباید پیدا کند**: نه شمارش دور، نه حساب بودجه، نه
سیاست ارجاع. فقط `run_task` را import می‌کند، سه قلاب بالا را می‌دهد، و نتیجه را رندر می‌کند.
هر منطقی که آنجا تکرار شود یک منبع حقیقت دوم می‌سازد که از اولی جدا می‌افتد.

قرارداد thread — این‌ها را نشکن:

- `run_task` در یک thread پس‌زمینه اجرا می‌شود؛ thread هرگز به `st.session_state` یا هیچ
  `st.*` دست نمی‌زند (هیچ‌کدام thread-safe نیستند و بدون script context استثنا می‌دهند).
- **دو صف** تنها کانال ارتباطی‌اند: `outbox` از thread به UI (رخدادها، سؤال‌ها، نتیجه‌ی نهایی)
  و `answers` از UI به thread (لیبل گزینه‌ی انتخاب‌شده). thread روی `answers.get()` بلاک
  می‌شود — که دقیقاً درست است: کنترلر وسط یک تصمیم است و نباید با حدس ادامه دهد.
- **`stop_flag` یک صف نیست**، یک latch است: باید حتی وقتی thread جای دیگری بلاک است هم
  خوانده شود.
- هر مسیر خروج از thread دقیقاً **یک پیام پایانی** می‌فرستد، وگرنه UI برای همیشه منتظر
  thread ای می‌ماند که مرده است.
- نمایش زنده = خالی‌کردن صف + `st.rerun()`، نه حلقه‌ای که رشته‌ی اصلی را قفل کند. وقتی سؤالی
  روی صفحه است polling متوقف می‌شود تا `st.radio` زیر دست کاربر ری‌رندر نشود.
- کلیک «توقف» علاوه بر `stop_flag.set()` یک پاسخ خالی هم در `answers` می‌گذارد؛ وگرنه thread
  تا سقف timeout روی سؤال بی‌پاسخ پارک می‌ماند.
- سقف انتظار برای پاسخ کاربر ۳۰۰ ثانیه است. بعد از آن thread با استثنا آزاد می‌شود تا تبی که
  کسی به آن برنگشته یک thread را برای همیشه نگه ندارد.
- **وضعیت کلیدهای API فقط بله/خیر** نمایش داده می‌شود — نه خود کلید، نه هیچ بخشی از آن،
  حتی به‌صورت ماسک‌شده.

## لاگ

هر اجرا یک فایل `runs/<run_id>.jsonl` می‌سازد. رویدادها: `run_start`، `manager_plan`،
`worker_output`، `critic_verdict`، `escalation`، `escalation_answer`، `budget_raised`،
`run_end` — هرکدام با هزینه‌ی همان مرحله.

## دستورها

```bash
make test        # self_check، بدون کلید API و بدون شبکه
make lint        # py_compile روی همه‌ی فایل‌ها
make run         # اجرای واقعی در ترمینال (هزینه دارد)
make ui          # رابط Streamlit روی همان run_task
```

`scripts/self_check.py` قبل از هر commit باید سبز باشد؛ بدون کلید و بدون شبکه اجرا می‌شود و
تبدیل‌کننده‌های اسکیما، حساب بودجه و هر سه تریگر ارجاع را با نقش‌های ساختگی تست می‌کند.

# 📌 Before proceeding with TASKS, make sure to read the [➡️ Setup Instructions](PIXI_SETUP.md) first!

---

# **Pixi Tasks: Development and Build Automation 🏗️**

Pixi Tasks is a system designed to automate different stages of a project’s development and build process. Instead of manually running configuration, build, testing, formatting, or application launch commands, you can configure all of these steps under a single interface. This approach helps:

- **Reduce repetitive operations** ⚙️
  - No more typing the same commands over and over.
- **Ensure sequential execution** 🔄
  - Tasks can depend on each other, ensuring the correct order (for example, config → build → run).
- **Speed up development** ⏩
  - Caching results and automating repeated actions makes workflows faster.

---

## Main Features of Pixi Tasks 🎯

### 1. Task Configuration 🏗️
All tasks are defined in the `pixi.toml` file, and there are two main ways to describe them:

#### **String Definition 📝**
A simple command is specified as a string:

```toml
[tasks]
build = "ninja -C .build"
```

#### **Object Definition with Parameters ⚙️**
This method allows you to add extra parameters, such as:
- `cmd` — the command to run
- `depends-on` — tasks that must be completed first
- `cwd` — the working directory
- `env` — environment variables
- `inputs` and `outputs` — for caching

**Example:**

```toml
[tasks]
configure = { cmd = "cmake -G Ninja -S . -B .build" }
build = { cmd = "ninja -C .build", depends-on = ["configure"] }
```

---

### 2. Task Dependencies 🔗
One of the key features of Pixi Tasks is the ability to set dependencies between tasks, effectively creating execution chains. For example, you can have **configure → build → run**:

```toml
[tasks]
configure = "cmake -G Ninja -S . -B .build"
build = { cmd = "ninja -C .build", depends-on = ["configure"] }
start = { cmd = ".build/bin/app", depends-on = ["build"] }
```

If one task fails (returns a non-zero code), subsequent tasks will not run, preventing further issues if something goes wrong.

---

### 3. Environment Variables 🌐
You can set environment variables for each task. This is handy for defining default values that can be overridden from the command line if needed.

**Example:**

```toml
[tasks]
echo = { cmd = "echo $ARGUMENT", env = { ARGUMENT = "hello" } }
```

Running the task without overriding:

```bash
pixi run echo
```

The output will be `hello`. If you override the variable:

```bash
ARGUMENT=world pixi run echo
```

You will see `world`.

---

### 4. Working Directory (`cwd`) 📂
Sometimes you might want commands to run in a specific directory—especially useful in structured projects. You can specify a working directory for each task using the `cwd` parameter.

**Example:**

```toml
[tasks]
bar = { cmd = "python bar.py", cwd = "scripts" }
```

---

### 5. Task Caching 🗄️
Pixi Tasks supports caching based on specified input and output files. This means:

- If the files in `inputs` haven’t changed and the result (e.g., a file from `outputs`) already exists, the command won’t run again.
- The system calculates checksums (fingerprints) for files and compares them with the previous run.

**Examples:**

```toml
[tasks]
run = { cmd = "python main.py", inputs = ["main.py"] }
download_data = { cmd = "curl -o data.csv https://example.com/data.csv", outputs = ["data.csv"] }
build = { cmd = "make", inputs = ["src/*.cpp", "include/*.hpp"], outputs = ["build/app.exe"] }
```

---

### 6. Built-in Shell Commands 🖥️
Pixi Tasks uses the `deno_task_shell`, which supports basic cross-platform shell commands such as:

- **File operations**: `cp`, `mv`, `rm`, `mkdir`
- **Information output**: `echo`, `cat`, `pwd`
- **Execution control**: `sleep`, `exit`, `unset`
- **Pipes and redirection**: `|`, `|&`, `>`, `>>`

Additionally, shell operators like `&&` (execute next command if previous succeeds) and `||` (execute next command if previous fails) are supported.

---

## Why Use Tasks? 🤔

1. **Streamlined workflow**
   - Group multiple commands into a single call. No need to memorize or repeat long command sequences.
2. **Guaranteed order of execution**
   - Dependencies ensure preliminary steps (like configuration) happen before main ones (like build or test).
3. **Fewer errors**
   - If a task fails, the process stops, preventing further issues and easing debugging.
4. **Performance boosts**
   - Caching avoids re-running expensive operations. This is particularly helpful for lengthy builds or tests.
5. **Standardization and scalability**
   - A single system for defining and running tasks simplifies maintenance, onboarding new team members, and setting up CI/CD pipelines.

---

With Pixi Tasks, you can 🚀 accelerate development, maintain a clean workflow, and reduce repetitive operations, all while ensuring your build and deployment processes run smoothly.

---

# Lint Tasks Setup 🚀

Here’s how to add tasks for **Black**, **Pylint**, and **mypy** via CLI using `pixi task add` and creating an alias to aggregate them:

## ✨ Step 1: Add the Black task

```bash
pixi task add black "black ."
```

This command adds a **black** task to your configuration (for example, `pyproject.toml`), which runs `black .` to format your code.

## 🔍 Step 2: Add the Pylint task

```bash
pixi task add pylint "pylint ."
```

This command adds a **pylint** task that runs `pylint .` for code quality checks.

## 🛠 Step 3: Add the mypy task

```bash
pixi task add mypy "mypy ."
```

This command adds a **mypy** task that runs `mypy .` for static type checking.

## 📦 Step 4: Create an alias for lint

To group these tasks under one command, run:

```bash
pixi task alias lint black pylint mypy
```

This creates a **lint** task that depends on **black**, **pylint**, and **mypy**. When you run the **lint** task, it executes all three tools in sequence.

## 🚀 Final Run

To run all tasks with a single command, use:

```bash
pixi run -e dev lint
```

We use the `-e dev` option because the required packages (e.g., **black**, **pylint**, **mypy**) are installed in a dedicated **dev** environment, ensuring they are available to run.

This will sequentially execute:
- **black** for code formatting.
- **pylint** for code quality.
- **mypy** for static type checking.

If any one of these tasks fails with a non-zero exit code, the remaining tasks will not run.

## 🎉 Running tasks individually

You can also run each tool separately in the **dev** environment. For example:

```bash
pixi run -e dev black
```

Likewise, simply replace `black` with `pylint` or `mypy` to run those tasks individually. Again, the `-e dev` ensures the commands are executed in the environment where the libraries are installed.

## 🗒 Example `pyproject.toml` settings

Below is an example snippet showing how your `pyproject.toml` might look with these tasks:

```toml
[tool.pixi.tasks]
black = "black ."
pylint = "pylint ."
mypy = "mypy ."
lint = { depends-on = ["black", "pylint", "mypy"] }
```


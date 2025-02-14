# 🚀 Installing Pixi

To install Pixi, run the following command:

```sh
curl -fsSL https://pixi.sh/install.sh | bash
```

This will download and install Pixi on your system.

# 🚀 Initializing a Project with Pixi

Pixi supports two manifest formats: `pyproject.toml` and `pixi.toml`. In this guide, we will use the `pyproject.toml` format as it is the most common for Python projects.

## 🏁 Getting Started

If you are already inside a project directory, you can initialize Pixi by running the following command:

```sh
pixi init --format pyproject
```
This command sets up Pixi within the existing project structure.

### 📂 Project Structure

After running the command, a project directory will be created with the following structure:

```
<project_name>/
├── pyproject.toml
└── src/
    └── <project_name>/
        └── __init__.py
```

- `pyproject.toml` – The main project configuration file.
- `src/` – The directory containing the source code.
- `<project_name>/__init__.py` – A file required to define the package.

Now your project is ready for further configuration and development using Pixi!

## 🔧 Additional Options

If you are initializing a completely new project, you can specify the project name during initialization:

```sh
pixi init my_project --format pyproject
```

This will create a `my_project` directory with the following structure:

```
my_project/
├── pyproject.toml
└── src/
    └── my_project/
        └── __init__.py
```

### 📁 When Should You Use the `src/` Directory?

### 1️⃣ When You Don't Need to Move Code into `src/`

If your project is already structured correctly and runs as expected, you can leave it as is.

✅ Your code is already organized into modules (e.g., `dataset_modules`, `losses`, `models`, `toolkit`).
✅ Pixi supports your existing structure and does not require `src/`.
✅ Python imports modules correctly (`import models`, `import toolkit`, etc.).

📌 Keep the current structure if everything works fine.

### 2️⃣ When You Should Switch to an `src/`-Based Structure

Moving code into `src/` can be beneficial if:

- You want to prevent implicit imports (e.g., running `pytest` in the root directory does not accidentally load modules).
- The project will be published as a package (better for `pip install -e .`).
- Pixi creates new projects with `src/` to follow a consistent structure.

📌 If you choose an `src/`-based structure, move your code as follows:

```sh
mkdir -p src/my_project
mv dataset_modules losses models toolkit src/my_project/
```

Then update `pyproject.toml`:

```toml
[tool.pixi.pypi-dependencies]
my_project = { path = "src/my_project", editable = true }
```

Note that this requires updating imports throughout your code (e.g., `import my_project.models` instead of `import models`).

### 3️⃣ Conclusion

❌ You don’t need to change the structure if everything works fine.
If Pixi does not require changes, it’s best to keep things as they are.

📌 Moving to `src/` is useful if:

- You want to avoid import issues (especially when testing).
- You plan to publish the project as a Python package.

🔹 If keeping the current structure → no changes needed.
🔹 If switching to `src/` → prepare to update import paths. 🚀

## 🐍 Specifying the Python Version

Pixi allows you to define the required Python version for your project in the `pyproject.toml` file. This ensures that the correct Python interpreter is used when setting up the environment.

To specify a required Python version, add the following line to your `pyproject.toml`:

```toml
[project]
requires-python = "==3.11"
```

This setting enforces the use of Python 3.11. If a different Python version is installed, Pixi will notify you of the mismatch.

### Why Define a Python Version?

- Ensures consistency across development environments.
- Prevents compatibility issues when using specific libraries.
- Helps reproduce the same setup across different machines.

After specifying the Python version, you can install the environment using:

```sh
pixi install
```

Pixi will ensure that the specified Python version is used when creating the environment.


## 🌐 Creating a Default Environment

To add a default environment to your project, run:

```sh
pixi project environment add default --solve-group default --force
```

🔍 **Command Breakdown:**

- `pixi project environment add` → Adds a new environment to `pyproject.toml`.
- `default` → Name of the new environment (can be changed, but we use `default`).
- `--solve-group default` → Defines that this environment belongs to a "solve-group" named `default`.
  - Solve-groups in Pixi allow grouping dependencies (e.g., dev, testing, production).
  - If multiple environments share the same solve-group, Pixi caches and reuses resolved dependencies, speeding up installation.

📌 After executing this command, `pyproject.toml` will contain:

```toml
[tool.pixi.environments]
default = { solve-group = "default" }
```

This is a basic environment that can be installed with:

```sh
pixi install --environment default
```

## 📦 Installing Packages in the Default Environment

Once the environment is set up, you can add packages using:

```sh
pixi add pytorch-lightning==2.50
pixi add optuna==4.20
```

This will install `pytorch-lightning` and `optuna` and add them to the `pyproject.toml` dependencies:

```toml
[tool.pixi.dependencies]
pytorch-lightning = "==2.5.0"
optuna = "==4.2.0"
```

If a package is not available in Conda, you can install it from PyPI using the `--pypi` flag:

```sh
pixi add --pypi cylimiter==0.4.2
```

This will install `cylimiter` and add it to the `pyproject.toml` dependencies:

```toml
dependencies = ["cylimiter==0.4.2"]
```

# 🌱 Configuring the Development Environment in Pixi

## 1. Adding Dependencies for the Development Environment (dev)

### Command:

```sh
pixi add --feature dev pylint ==3.3.4
```

### What Happens:
- `pixi add` – Adds a dependency to the project.
- `--feature dev` – Specifies that the dependency is added to a special group (feature) named `dev`, which is meant for development dependencies (e.g., linting tools). This follows the principles outlined in **PEP 735**, which introduces standardized dependency groups for Python projects.
- `pylint ==3.3.4` – Specifies the package name and version.

### Result:
This command updates the `pyproject.toml` file, adding the `dev` feature section:

```toml
[tool.pixi.feature.dev.dependencies]
pylint = "==3.3.4"
```

Now, `pylint` version `3.3.4` is included in the `dev` feature, which will only be used in environments where this feature is enabled.

---

## 2. Configuring the `dev` Environment

### Command:

```sh
pixi project environment add dev --feature dev --solve-group default
```

### What Happens:
- `pixi project environment add dev` – Adds a new environment named `dev`.
- `--feature dev` – Ensures that this environment includes dependencies from the `dev` feature (e.g., `pylint`).
- `--solve-group default` – Uses the `default` dependency resolution group, meaning both standard and `dev` dependencies will be installed.

### Result:
This command updates `pyproject.toml` by adding the following:

```toml
[tool.pixi.environments]
default = { solve-group = "default" }
dev = { features = ["dev"], solve-group = "default" }
```

- `default` – The base environment using the `default` dependency resolution group.
- `dev` – A new environment that includes both the standard dependencies and those from the `dev` feature (e.g., `pylint`).

---

## 3. Summary of the Workflow

### Step 1: Add a Development Dependency
The command `pixi add --feature dev pylint ==3.3.4` registers `pylint` under the `dev` feature in `pyproject.toml`, following **PEP 735**, which defines dependency groups.

### Step 2: Configure the Development Environment
The command `pixi project environment add dev --feature dev --solve-group default` creates a `dev` environment that includes both standard dependencies and those from the `dev` feature.

### Benefits:
- **Separation of dependencies** – Easily distinguish between development and production dependencies.
- **Environment-specific tools** – Development environments can have additional tools (like `pylint`) without affecting the main environment.
- **Flexibility** – Users can activate different environments as needed.
- **PEP 735 Compliance** – This approach ensures structured and standardized dependency management according to modern Python packaging recommendations.

By following this approach, you ensure a clean and maintainable project setup with clearly defined environments for different use cases.


## 📥 Installing and Running the dev Environment

To install dependencies for the `dev` environment, run:

```sh
pixi install --environment dev
```

Then, you can execute `pylint` using:

```sh
pixi run --environment dev pylint .
```

# 🤝 Activating a Pixi Environment

To use a Pixi environment, it must be activated. Activation ensures that the correct dependencies and configurations are applied within the environment, preventing conflicts with system-wide installations.

## ⚙️ Activation Methods

There are several ways to activate a Pixi environment, depending on your needs:

1. **Using `pixi shell`**: Opens a new shell session with the environment activated. This method is useful when you want to work interactively within the environment.

   ```sh
   pixi shell
   ```

2. **Using `pixi run`**: Runs a command within the activated environment without launching a new shell session. This is particularly useful for executing scripts or single commands.

   ```sh
   pixi run python script.py
   ```

3. **Using `pixi shell-hook`**: Outputs the necessary commands to activate the environment manually in your current shell session. This is helpful when integrating Pixi into automated workflows.

   ```sh
   pixi shell-hook
   ```

For more information, refer to the [official Pixi documentation](https://pixi.sh/latest).


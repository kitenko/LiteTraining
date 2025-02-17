# 🤔 Why Pixi?

Pixi is a powerful environment manager and packaging solution for Python that simplifies environment creation, dependency management, and environment activation. By using conda or PyPI-based dependencies, Pixi provides a consistent and flexible approach for development, testing, and production workflows. In short, it helps keep your project clean, reproducible, and easy to maintain. Additionally, Pixi automatically generates a lock file that ensures you can precisely reproduce your environment across different systems.

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

## Working with Extras in Pixi 🚀

When using PyPI dependencies in Pixi, you can specify additional options through the `extras` parameter. This allows you to install not only the core package but also optional dependencies that extend its functionality. For example, to install `black` with Jupyter support, you need to specify the extra option `"jupyter"`, since the corresponding package might not be available via Conda.

---

### Example Configuration 🔧

To install `black` with Jupyter support, add the following entry in your configuration file (e.g., `pyproject.toml`) under the section for dev PyPI dependencies:

```toml
[tool.pixi.feature.dev.pypi-dependencies]
black = { version = ">=25.1.0,<26", extras = ["jupyter"] }
```

---

### Explanation of the Fields 📖

- **`version`**: Sets the version of the package. In this example, `black` will be installed with a version that satisfies the condition `>=25.1.0,<26`.
- **`extras`**: A list of additional options to install along with the core package. Here, `"jupyter"` ensures that `black` installs the dependencies required to work with Jupyter Notebook.

---

## When to Use Extras? 🤔

### ✅ Extended Functionality
If a package has optional features that are not needed for basic functionality, `extras` allow you to install these capabilities on demand.

### 📦 Missing Packages in Conda
If the desired package or its extended functionality isn’t available via Conda, `extras` let you leverage PyPI to install the necessary dependencies.

### 📌 Dependency Management
Using `extras`, you can centrally manage optional dependencies for specific environments (e.g., a dev environment that requires enhanced functionality).


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

# A Brief Overview of pixi.lock 🔒⚙️

## What is it?
**pixi.lock** is a file that captures the current environment and all installed packages (with their metadata). Think of it as a "lock" ensuring complete reproducibility.

## Why do you need it?
1. **Preserves State**: Lets you save and restore a working environment configuration.
2. **Speeds Development**: Simplifies collaboration and makes switching between versions easier.
3. **Flexibility**: Effortlessly roll back to a previous environment or transfer it to a new machine.

## How is it generated and used?
* **Automatically created** whenever you install a package (after the "solve" step).
* **Do not edit it manually**: It's best to keep it under version control (e.g., Git) to track changes.
* Whenever dependencies in `pixi.toml` or `pyproject.toml` change, the lock file automatically updates with commands like: `pixi install`, `pixi run`, `pixi shell`, `pixi add`, `pixi remove`, etc.

## Additional Parameters
* `--frozen` and `--locked`: These flags dictate how and whether `pixi.lock` updates during installation and usage.
  - **`--frozen`**: If any discrepancies occur, it won't update the lock file, using it "as is" and installing exactly what is specified.
  - **`--locked`**: Checks if the lock file is current; if it's out of sync with the manifest, the process stops, leaving `pixi.lock` unchanged.

## Versioning
* The **version** in `pixi.lock` shows compatibility with your current Pixi version.
* Pixi is backward compatible with older lock files, but older versions of Pixi won't recognize lock files created by newer Pixi releases.

## File Size
* Can become fairly large because it stores extensive package metadata.
* Still smaller than Docker images and faster to fetch than unnecessary packages.

## Removal
* `rm pixi.lock` simply removes the "lock". The environment will be re-solved the next time a command requires it.

---

Use **pixi.lock** if you need a fast, reliable way to recreate your environment at any moment ✨

# System Requirements in pixi 🚀💻

## General Purpose of System Requirements 🛠️🔑

System requirements in **pixi** help you define the minimum specs (OS, kernel, glibc, CUDA, etc.) needed to install and run your project environment. If your machine doesn’t meet these specs, `pixi run` fails because the environment can’t function correctly.

When **pixi** resolves dependencies, it combines:
1. **Default requirements** (tied to the OS).
2. **Custom requirements** (in `[system-requirements]`).

This ensures the environment is compatible with your hardware. 🖥️

---

## Examples of Standard System Requirements 🏷️💡

For instance, on Linux:

```toml
[system-requirements]
linux = "4.18"
libc = { family = "glibc", version = "2.28" }
```

- Linux kernel ≥ 4.18
- glibc ≥ 2.28

Other platforms (Windows, `osx-64`, `osx-arm64`) have their own defaults.

---

## Configuring System Requirements ⚙️📝

### 1. Adapting to Specific Systems 💻✅

If your system doesn’t match the defaults (e.g., older Linux kernel), override them:

```toml
[system-requirements]
linux = "4.12.14"
```

### 2. Using CUDA 🔥🚀

Specify CUDA in `[system-requirements]`:

```toml
[system-requirements]
cuda = "12"
```

This tells **pixi** which CUDA version is available. If a package needs `__cuda >= 12`, pixi resolves it.

### 3. Per-Environment Requirements 🌐🔧

You can set requirements for features:

```toml
[feature.cuda.system-requirements]
cuda = "12"

[environments]
cuda = ["cuda"]
```

This way, you can have separate environments for CUDA or CPU-only.

### 4. Environment Variables 🌍🔀

To override detected system requirements:

- **`CONDA_OVERRIDE_CUDA`** sets the CUDA version.
- **`CONDA_OVERRIDE_GLIBC`** sets the glibc version.
- **`CONDA_OVERRIDE_OSX`** sets the macOS version.

Use these if your system reports different values than you want.

---

## Integration with Project Settings 📦🤝

In `[tool.pixi.system-requirements]`, define the required CUDA:

```toml
[tool.pixi.system-requirements]
cuda = "12.4"
```

If you need to pin versions further:

```toml
[tool.pixi.dependencies]
cuda-version = "==12.4"
```

---

## Example Project Configuration 📝✨

```toml
[project]
name = "app"
requires-python = "== 3.11"
version = "0.1.0"
dependencies = ["cylimiter==0.4.2", "jsonargparse[signatures]", "tensorboard==2.19.0"]

[build-system]
build-backend = "hatchling.build"
requires = ["hatchling"]

[tool.pixi.project]
channels = ["conda-forge"]
platforms = ["linux-64"]

[tool.pixi.system-requirements]
cuda = "12.4"

[tool.pixi.pypi-dependencies]
app = { path = ".", editable = true }

[tool.pixi.tasks]

[tool.pixi.environments]
default = { solve-group = "default" }
dev = { features = ["dev"], solve-group = "default" }

[tool.pixi.dependencies]
pytorch-gpu = "==2.5.1"
torchvision = "==0.20.1"
torchaudio = "==2.5.1"
pytorch-lightning = "==2.5.0"
optuna = "==4.2.0"
datasets = "==3.2.0"
transformers = "==4.48.3"
python-dotenv = "==1.0.1"
albumentations = "==2.0.4"
scikit-learn = "==1.6.1"
seaborn = "==0.13.2"

[tool.pixi.feature.dev.dependencies]
pylint = "==3.3.4"
black = "==25.1.0"
pre-commit = "==4.1.0"
mypy = "==1.15.0"
types-pyyaml = "==6.0.12.20241230"
pytest-mock = "==3.14.0"
```

---

## System Requirements Summary ✅🎉

- **Purpose:** Ensure the project environment installs and runs correctly on the target system.
- **Default vs. Custom:** Overwrite defaults in `[system-requirements]` if needed.
- **CUDA Usage:** `cuda = "12.4"` in `[tool.pixi.system-requirements]` ensures the correct CUDA.
- **Separate Environments:** Configure different features for CPU/GPU.
- **Env Var Overrides:** `CONDA_OVERRIDE_CUDA` etc. let you override detection.

These tools keep your project stable and compatible! 💯

---

## PyTorch Integration Guide 🔥🧠

Learn how to integrate **PyTorch** with **pixi**:

1. **Conda-forge** (Recommended)
2. **PyPI** (via pixi’s uv integration)

Pick whichever fits your needs.

---

### PyTorch & System Requirements 🏆

PyTorch often requires CUDA. By setting `system-requirements.cuda = "12"`, you let **pixi** know your system supports CUDA 12. If that’s missing, you might get CPU-only installs.

---

## Installing from Conda-forge 🏗️

Use the conda-forge channel for community-maintained builds. You can also specify a `cuda-version` to pin the CUDA version.

### Minimal Example

```toml
[project]
name = "pytorch-conda-forge"

[tool.pixi.project]
channels = ["https://prefix.dev/conda-forge"]
platforms = ["linux-64"]

[tool.pixi.system-requirements]
cuda = "12.0"

[tool.pixi.dependencies]
pytorch-gpu = "*"
```

Or pin CUDA:

```toml
[tool.pixi.dependencies]
pytorch-gpu = "*"
cuda-version = "12.6"
```

### Splitting GPU and CPU Environments

```toml
[project]
name = "pytorch-conda-forge"

[tool.pixi.project]
channels = ["https://prefix.dev/conda-forge"]
platforms = ["linux-64"]

[tool.pixi.feature.gpu.system-requirements]
cuda = "12.0"

[tool.pixi.feature.gpu.dependencies]
cuda-version = "12.6"
pytorch-gpu = "*"

[tool.pixi.feature.cpu.dependencies]
pytorch-cpu = "*"

[tool.pixi.environments]
cpu = ["cpu"]
default = ["gpu"]
```

Run:

```bash
pixi run --environment cpu python -c "import torch; print(torch.cuda.is_available())"
pixi run -e gpu python -c "import torch; print(torch.cuda.is_available())"
```

---

## Installing from PyPI 🌐

Using **uv** integration, you can get PyTorch from **PyPI**.

### Don’t Mix Conda & PyPI if Dependent

If a Conda package depends on `torch`, everything must come from Conda. If a PyPI package depends on `torch`, everything must come from PyPI. Mixing them can cause conflicts.

### PyTorch Indexes 💾

PyTorch uses custom indexes:
- CPU: `https://download.pytorch.org/whl/cpu`
- CUDA 11.8: `.../cu118`
- CUDA 12.1: `.../cu121`
- CUDA 12.4: `.../cu124`
- ROCm6: `.../rocm6.2`

#### Example

```toml
[project]
name = "pytorch-pypi"
requires-python = ">= 3.11,<3.13"

[tool.pixi.project]
channels = ["https://prefix.dev/conda-forge"]
platforms = ["osx-arm64", "linux-64", "win-64"]

[tool.pixi.pypi-dependencies]
torch = { version = ">=2.5.1", index = "https://download.pytorch.org/whl/cu124" }
torchvision = { version = ">=0.20.1", index = "https://download.pytorch.org/whl/cu124" }

[tool.pixi.target.osx.pypi-dependencies]
# For macOS, use CPU
torch = { version = ">=2.5.1", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = ">=0.20.1", index = "https://download.pytorch.org/whl/cpu" }
```

### Multiple Environments for CPU/GPU

```toml
[project]
name = "pytorch-pypi-envs"
requires-python = ">= 3.11,<3.13"

[tool.pixi.project]
channels = ["https://prefix.dev/conda-forge"]
platforms = ["linux-64", "win-64"]

[tool.pixi.feature.gpu]
system-requirements = { cuda = "12.0" }

[tool.pixi.feature.gpu.pypi-dependencies]
torch = { version = ">=2.5.1", index = "https://download.pytorch.org/whl/cu124" }
torchvision = { version = ">=0.20.1", index = "https://download.pytorch.org/whl/cu124" }

[tool.pixi.feature.cpu.pypi-dependencies]
torch = { version = ">=2.5.1", index = "https://download.pytorch.org/whl/cpu" }
torchvision = { version = ">=0.20.1", index = "https://download.pytorch.org/whl/cpu" }

[tool.pixi.environments]
gpu = { features = ["gpu"] }
default = { features = ["cpu"] }
```

```bash
pixi run --environment cpu python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pixi run -e gpu python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### macOS + CUDA?

macOS doesn’t support CUDA. If you’re on macOS but want to resolve CUDA wheels for another platform, you can hit resolution issues. For now, do that on a system that supports CUDA.

---

## Troubleshooting 🛠️

### Test Installation ⚗️

```bash
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Check pixi’s Detected CUDA Info 🔎

```bash
pixi info
```

If `__cuda` is missing:

```bash
nvidia-smi
```

To check the CUDA toolkit:

```bash
pixi run nvcc --version
```

### Common Pitfalls 😅

- **Mixing Channels**: Using multiple Conda channels can cause conflicts. Stick to conda-forge if you can.
- **Mixing Conda & PyPI**: If PyTorch is from PyPI, anything that depends on `torch` must also come from PyPI.
- **GPU Version Not Installing**:
  - Make sure `system-requirements.cuda` is set.
  - Pin your CUDA version if needed.
- **ABI/Platform Mismatch**:
  - Ensure your Python version (and OS) match the wheels.
  - If you see errors about unsupported tags, choose another wheel or Python version.

With the right settings, your PyTorch installation should work smoothly! 🔥🚀



For more information, refer to the [official Pixi documentation](https://pixi.sh/latest).


@echo off
setlocal

echo Creating directories...

for %%D in (
    configs
    scripts
    tests
    outputs\models
    outputs\runs
    outputs\metrics
    outputs\figures
    report
    src\cartpole_rl
    src\cartpole_rl\simulation
    src\cartpole_rl\controllers
    src\cartpole_rl\envs
    src\cartpole_rl\training
    src\cartpole_rl\analysis
    src\cartpole_rl\plotting
) do (
    if not exist "%%D" mkdir "%%D"
)

echo Creating files...

for %%F in (
    README.md
    requirements.txt
    requirements-dev.txt
    pyproject.toml
    Makefile
    .gitignore
    configs\plant.yaml
    configs\nominal_controller.yaml
    configs\sac.yaml
    configs\shield.yaml
    configs\evaluation.yaml
    scripts\smoke_test.py
    scripts\train.py
    scripts\evaluate.py
    scripts\make_figures.py
    tests\test_dynamics.py
    tests\test_integrator.py
    tests\test_lqr.py
    tests\test_environment.py
    tests\test_shield.py
    tests\test_reproducibility.py
    report\main.tex
    src\cartpole_rl\__init__.py
    src\cartpole_rl\config.py
    src\cartpole_rl\types.py
    src\cartpole_rl\simulation\__init__.py
    src\cartpole_rl\simulation\dynamics.py
    src\cartpole_rl\simulation\integrators.py
    src\cartpole_rl\simulation\simulator.py
    src\cartpole_rl\controllers\__init__.py
    src\cartpole_rl\controllers\base.py
    src\cartpole_rl\controllers\energy_shaping.py
    src\cartpole_rl\controllers\lqr.py
    src\cartpole_rl\controllers\nominal.py
    src\cartpole_rl\controllers\residual.py
    src\cartpole_rl\controllers\lyapunov_shield.py
    src\cartpole_rl\envs\__init__.py
    src\cartpole_rl\envs\residual_cartpole.py
    src\cartpole_rl\training\__init__.py
    src\cartpole_rl\training\train_sac.py
    src\cartpole_rl\training\callbacks.py
    src\cartpole_rl\analysis\__init__.py
    src\cartpole_rl\analysis\rollouts.py
    src\cartpole_rl\analysis\fixed_points.py
    src\cartpole_rl\analysis\jacobians.py
    src\cartpole_rl\analysis\basin.py
    src\cartpole_rl\analysis\lyapunov.py
    src\cartpole_rl\analysis\mass_sweep.py
    src\cartpole_rl\analysis\metrics.py
    src\cartpole_rl\plotting\__init__.py
    src\cartpole_rl\plotting\figures.py
) do (
    if not exist "%%F" type nul > "%%F"
)

echo Creating Python virtual environment...
py -3.11 -m venv .venv

echo Initializing Git...
git init

echo.
echo Project structure created successfully.
echo Activate the environment with:
echo     .venv\Scripts\activate
echo.
pause
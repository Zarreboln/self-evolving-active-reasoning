# Source before running anything in mb/. Defaults are bath's; ~/.mb_site.sh (sourced first) can override.
export JAVA_HOME=${JAVA_HOME:-$HOME/opt/jdk17}
export PATH=$JAVA_HOME/bin:${MB_VENV:-$HOME/venvs/mbench}/bin:$PATH
export MBENCH_DATA=${MBENCH_DATA:-/data/team-bath/tmp_l/mbench_data}
export ALFWORLD_DATA=$MBENCH_DATA/alfworld
export PYTHONPATH=$HOME/code/multibench${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false

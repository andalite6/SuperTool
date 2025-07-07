#!/usr/bin/env bash
set -e

# Create external directory
mkdir -p external

# Clone repositories at specific commits

git clone https://github.com/NVIDIA/garak.git external/garak
cd external/garak && git checkout 37d046be30883016d99992cd65d8308b940fa701 && cd ../../

git clone https://github.com/confident-ai/deepteam.git external/deepteam
cd external/deepteam && git checkout 7c870c369c2b01a8181b746291ee7948822c9d6e && cd ../../

# Red-Teaming-Toolkit pinned commit
git clone https://github.com/infosecn1nja/Red-Teaming-Toolkit.git external/Red-Teaming-Toolkit
cd external/Red-Teaming-Toolkit && git checkout 14dc0dbf77c2259819cd1d555ba9c97f9264e0a8 && cd ../../

# Others: replace <latest_commit_hash> accordingly
git clone https://github.com/A-poc/RedTeam-Tools.git external/RedTeam-Tools
git clone https://github.com/Azure/PyRIT.git external/PyRIT
git clone https://github.com/bethgelab/meerkat.git external/meerkat
git clone https://github.com/redcanaryco/atomic-red-team.git external/atomic-red-team
git clone https://github.com/promptfoo/promptfoo.git external/promptfoo
git clone https://github.com/mitre/caldera.git external/caldera

# Ensure msfrpc gem installed for Metasploit
# gem install msfrpc-client

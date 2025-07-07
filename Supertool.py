import os
import streamlit as st
from orchestrator import RedTeamOrchestrator

# Initialize the orchestrator with registered adapters
orchestrator = RedTeamOrchestrator()

st.set_page_config(page_title="AI Red Team Super Tool", layout="wide")
st.title("🔴 AI Red Team Super Tool")

# Documentation download
st.sidebar.subheader("Documentation")
if os.path.exists("docs/Unified_AI_Assistant_System_Instructions_v2.pdf"):
    with open("docs/Unified_AI_Assistant_System_Instructions_v2.pdf", "rb") as f:
        pdf_bytes = f.read()
    st.sidebar.download_button("Download System Instructions", data=pdf_bytes,
                               file_name="Unified_AI_Assistant_System_Instructions_v2.pdf")

st.sidebar.markdown("---")
# Attack configuration
st.sidebar.subheader("Attack Configuration")
model_choice = st.sidebar.selectbox("Model", ["gpt-4", "bert-base-uncased", "custom"])
adapter_choice = st.sidebar.selectbox("Toolkit", orchestrator.available_adapters())
input_text = st.sidebar.text_area("Input text for attack", height=150)

if st.sidebar.button("🚀 Run Attack"):
    with st.spinner("Running attack..."):
        try:
            results = orchestrator.run_attack(adapter_choice, model_choice, input_text)
            st.success("✅ Attack completed!")
            # Convert results to DataFrame and display
            import pandas as pd
            df = pd.json_normalize(results)
            st.dataframe(df)
            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", data=csv,
                               file_name=f"{adapter_choice}_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error running {adapter_choice}: {e}")

# File: orchestrator.py
class RedTeamOrchestrator:
    def __init__(self):
        from adapters.garak_adapter import GarakAdapter
        from adapters.deepteam_adapter import DeepTeamAdapter
        from adapters.redteaming_toolkit_adapter import RTKAdapter
        from adapters.redteam_tools_adapter import RedTeamToolsAdapter
        from adapters.pyrt_adapter import PyRITAdapter
        from adapters.foolbox_adapter import FoolboxAdapter
        from adapters.meerkat_adapter import MeerkatAdapter
        from adapters.atomic_rt_adapter import AtomicRTAdapter
        from adapters.promptfoo_adapter import PromptFooAdapter
        from adapters.metasploit_adapter import MetasploitAdapter
        from adapters.caldera_adapter import CalderaAdapter

        self.adapters = {
            'garak': GarakAdapter(),
            'deepteam': DeepTeamAdapter(),
            'rtk': RTKAdapter(),
            'redteam_tools': RedTeamToolsAdapter(),
            'pyrt': PyRITAdapter(),
            'foolbox': FoolboxAdapter(),
            'meerkat': MeerkatAdapter(),
            'atomic_rt': AtomicRTAdapter(),
            'promptfoo': PromptFooAdapter(),
            'metasploit': MetasploitAdapter(),
            'caldera': CalderaAdapter(),
        }

    def available_adapters(self):
        return sorted(self.adapters.keys())

    def run_attack(self, adapter_name, model, input_data):
        adapter = self.adapters.get(adapter_name)
        if not adapter:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        return adapter.attack(model, input_data)

# File: adapters/garak_adapter.py
import sys
from pathlib import Path
GARAK_PATH = Path(__file__).parent.parent / "external" / "garak"
sys.path.append(str(GARAK_PATH))
from garak.attack import AttackRunner
class GarakAdapter:
    def __init__(self):
        self.runner = AttackRunner()
    def attack(self, model, input_data):
        return self.runner.run(model_name=model, prompt=input_data)

# File: adapters/deepteam_adapter.py
import sys
from pathlib import Path
DEEPTEAM_PATH = Path(__file__).parent.parent / "external" / "deepteam"
sys.path.append(str(DEEPTEAM_PATH))
from deepteam.run import run_attack as deepteam_run
class DeepTeamAdapter:
    def attack(self, model, input_data):
        return deepteam_run(model_name=model, text=input_data)

# File: adapters/redteaming_toolkit_adapter.py
import sys
from pathlib import Path
RTK_PATH = Path(__file__).parent.parent / "external" / "Red-Teaming-Toolkit"
sys.path.append(str(RTK_PATH))
from toolkit import run
class RTKAdapter:
    def attack(self, model, input_data):
        return run(model_name=model, text=input_data)

# File: adapters/redteam_tools_adapter.py
import sys
from pathlib import Path
RT_TOOLS_PATH = Path(__file__).parent.parent / "external" / "RedTeam-Tools"
sys.path.append(str(RT_TOOLS_PATH))
from redteam_tools import execute_attack
class RedTeamToolsAdapter:
    def attack(self, model, input_data):
        return execute_attack(model_name=model, text=input_data)

# File: adapters/pyrt_adapter.py
import sys
from pathlib import Path
PYRIT_PATH = Path(__file__).parent.parent / "external" / "PyRIT"
sys.path.append(str(PYRIT_PATH))
from pyrt import run_rit
class PyRITAdapter:
    def attack(self, model, input_data):
        return run_rit(model_name=model, prompt=input_data)

# File: adapters/foolbox_adapter.py
import sys
from pathlib import Path
FOOLBOX_PATH = Path(__file__).parent.parent / "external" / "foolbox"
sys.path.append(str(FOOLBOX_PATH))
from foolbox import PyTorchModel, samples, accuracy, attacks
class FoolboxAdapter:
    def attack(self, model, input_data):
        fmodel = PyTorchModel(model)
        images, labels = samples(fmodel, dataset='imagenet', batchsize=1)
        adv = attacks.FGSM()(fmodel, images, labels)
        succ = accuracy(fmodel, adv, labels)
        return {'success_rate': float(succ)}

# File: adapters/meerkat_adapter.py
import sys
from pathlib import Path
MEERKAT_PATH = Path(__file__).parent.parent / "external" / "meerkat"
sys.path.append(str(MEERKAT_PATH))
from meerkat import Meerkat
class MeerkatAdapter:
    def __init__(self): self.attacker = Meerkat()
    def attack(self, model, input_data): return self.attacker.run(model_name=model, prompt=input_data)

# File: adapters/atomic_rt_adapter.py
import sys
from pathlib import Path
ART_PATH = Path(__file__).parent.parent / "external" / "atomic-red-team"
sys.path.append(str(ART_PATH))
from atomic_red_team import AtomicRunner
class AtomicRTAdapter:
    def __init__(self): self.runner = AtomicRunner()
    def attack(self, model, input_data): return self.runner.run(model_name=model, text=input_data)

# File: adapters/promptfoo_adapter.py
import sys
from pathlib import Path
PROMPTFOO_PATH = Path(__file__).parent.parent / "external" / "promptfoo"
sys.path.append(str(PROMPTFOO_PATH))
from promptfoo import PromptFoo
class PromptFooAdapter:
    def __init__(self): self.pf = PromptFoo()
    def attack(self, model, input_data): return self.pf.run(model_name=model, prompt=input_data)

# File: adapters/metasploit_adapter.py
import sys
from pathlib import Path
MSF_PATH = Path(__file__).parent.parent / "external" / "metasploit-framework"
sys.path.append(str(MSF_PATH))
from msfrpc import MsfRpcClient
class MetasploitAdapter:
    def __init__(self): self.client = MsfRpcClient('password')
    def attack(self, model, input_data):
        module = self.client.modules.use('exploit', 'auxiliary/scanner/http/http_version')
        module['RHOSTS'] = input_data
        return module.execute()

# File: adapters/caldera_adapter.py
import sys
from pathlib import Path
CALDERA_PATH = Path(__file__).parent.parent / "external" / "caldera"
sys.path.append(str(CALDERA_PATH))
from app import conf, create_service
class CalderaAdapter:
    def __init__(self): self.service = create_service(conf)
    def attack(self, model, input_data):
        adv = self.service.adversary_svc.create_adversary(name="auto", description="via UI")
        return self.service.operation_svc.execute(adv.id)

# File: requirements.txt
streamlit
pandas
torch
foolbox
msfrpc
# Plus external dependencies for: garak, deepteam, Red-Teaming-Toolkit, RedTeam-Tools, PyRIT, meerkat, atomic-red-team, promptfoo, caldera

# Project structure:
# super-red-team-tool/
# ├── app.py
# ├── orchestrator.py
# ├── adapters/
# ├── external/
# ├── docs/
# ├── requirements.txt
# └── README.md

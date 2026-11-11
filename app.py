import sys
import importlib.util
import streamlit as st

st.write("실제 실행 중인 Python:", sys.version)
st.write("Python 실행 위치:", sys.executable)

for package in ["joblib", "sklearn", "rdkit", "numpy"]:
    spec = importlib.util.find_spec(package)
    st.write(
        package,
        "설치됨" if spec is not None else "찾을 수 없음"
    )

st.stop()  # 진단 정보를 확인하기 위해 여기서 잠시 중단
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Crippen, rdMolDescriptors,
    rdFingerprintGenerator, Draw
)

st.set_page_config(
    page_title="피부 감작성 예측",
    page_icon="🧪",
    layout="centered"
)

# 저장된 모델 불러오기
@st.cache_resource
def load_bundle():
    path = Path(__file__).parent / "model.joblib"
    return joblib.load(path)

try:
    bundle = load_bundle()
    model = bundle["model"]
except Exception as error:
    st.error("모델을 불러오지 못했습니다. 모델 파일과 설치 버전을 확인하세요.")
    st.code(str(error))
    st.stop()

# 학습 코드와 동일한 계산 순서
descriptor_names = [
    "MolWt", "LogP", "TPSA", "HBD", "HBA",
    "RotBonds", "RingCount",
    "Aldehyde", "MichaelMotif", "Epoxide", "Isocyanate"
]

patterns = [
    Chem.MolFromSmarts(s) for s in [
        "[$([CX3H1](=O)[#6]),$([CX3H2]=O)]",
        "[CX3]=[CX3]-[CX3](=[OX1])",
        "[OX2;r3]1[CX4;r3][CX4;r3]1",
        "[NX2]=[CX2]=[OX1]"
    ]
]

fp_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    includeChirality=False
)

def make_features(smiles):
    smiles = smiles.strip()
    if not smiles:
        raise ValueError("SMILES를 입력해주세요.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("해석할 수 없는 SMILES입니다.")

    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("분리된 여러 성분은 현재 입력 대상에서 제외합니다.")

    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        raise ValueError("현재 모델은 탄소를 포함하는 유기화합물 대상입니다.")

    values = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol, includeSandP=False),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True),
        rdMolDescriptors.CalcNumRings(mol),
    ]

    values += [
        int(mol.HasSubstructMatch(pattern))
        for pattern in patterns
    ]

    fingerprint = fp_generator.GetFingerprintAsNumPy(mol)
    vector = np.concatenate([values, fingerprint]).astype(float)

    if not np.isfinite(vector).all():
        raise ValueError("분자 특성을 정상적으로 계산하지 못했습니다.")

    return mol, vector

# 웹 화면
st.title("🧪 피부 감작성 예측")
st.write("분자구조를 이용한 연구용 사전 선별 프로그램")
st.caption("SVM · RDKit 분자 특성 · Morgan fingerprint")

st.info(
    "현재 모델은 연구·시연 단계입니다. "
    "분류 결과만으로 물질의 안전성을 판단할 수 없습니다."
)

with st.form("prediction_form"):
    smiles = st.text_input(
        "SMILES 입력",
        placeholder="예: C=O",
        max_chars=2000
    )
    submitted = st.form_submit_button("예측하기")

if submitted:
    try:
        mol, vector = make_features(smiles)
        prediction = int(model.predict(vector.reshape(1, -1))[0])

        st.subheader("모델 분류 결과")
        if prediction == 1:
            st.write("**감작성 쪽으로 예측**")
        else:
            st.write("**비감작성 쪽으로 예측**")

        st.image(Draw.MolToImage(mol, size=(450, 280)))
        st.write("해석된 SMILES")
        st.code(Chem.MolToSmiles(mol), language=None)

        key = Chem.MolToInchiKey(mol)
        if key in bundle.get("training_keys", []):
            st.caption("이 구조는 모델 학습에 사용된 자료에 포함되어 있습니다.")

        with st.expander("계산한 분자 특성 보기"):
            table = pd.DataFrame({
                "분자 특성": descriptor_names,
                "계산값": vector[:11]
            })
            st.dataframe(table.round(3))

    except ValueError as error:
        st.warning(str(error))
    except Exception:
        st.error("예측 실행 오류입니다. 서버 로그를 확인해주세요.")

with st.expander("모델 검증 정보"):
    scores = np.asarray(bundle.get("cv_scores", []), dtype=float)
    if scores.size:
        st.write(
            f"5-fold Balanced accuracy: "
            f"{scores.mean():.3f} ± {scores.std():.3f}"
        )
    st.write("개별 입력 물질의 예측 신뢰도를 나타내는 값은 아닙니다.")
    st.write("구조 유사도와 적용 가능 범위 검사는 아직 포함하지 않았습니다.")

st.divider()
st.caption(
    "본 결과는 개인의 알레르기 발생 여부나 화장품의 안전성을 "
    "확정하지 않습니다. 농도, 제형, 노출 조건 및 개인차는 "
    "본 모델에 반영되지 않았습니다."
)

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('gruut')
datas += collect_data_files('gruut_ipa')
datas += collect_data_files('gruut_lang_en')


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['gruut_ipa','gruut_lang_en'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='styletts2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='styletts2',
)

import os
import shutil
shutil.copy2('config.yaml','dist/styletts2/config.yaml')
shutil.copy2('ManjaroMix.yaml','dist/styletts2/ManjaroMix.yaml')
shutil.copy2('episodes_labels_index.json',
    'dist/styletts2/episodes_labels_index.json')
shutil.copy2('horsewords.clean',
    'dist/styletts2/horsewords.clean')
os.makedirs('dist/styletts2/Models', exist_ok=True)
os.makedirs('dist/styletts2/Utils/PLBERT', exist_ok=True)
os.makedirs('dist/styletts2/Utils/JDC', exist_ok=True)
os.makedirs('dist/styletts2/Utils/ASR', exist_ok=True)
shutil.copy2('Utils/PLBERT/config.yml','dist/styletts2/Utils/PLBERT/config.yml')
shutil.copy2('Utils/ASR/config.yml','dist/styletts2/Utils/ASR/config.yml')
shutil.copy2('Utils/ASR/epoch_00080.pth','dist/styletts2/Utils/ASR/epoch_00080.pth')
shutil.copy2('Utils/JDC/bst.t7','dist/styletts2/Utils/JDC/bst.t7')
shutil.copy2('Utils/JDC/bst_rmvpe.t7','dist/styletts2/Utils/JDC/bst_rmvpe.t7')
shutil.copy2('Utils/PLBERT/step_1000000.t7','dist/styletts2/Utils/PLBERT/step_1000000.t7')
shutil.copytree('Models/Multi0_40_24k','dist/styletts2/Models/Multi0_40_24k')
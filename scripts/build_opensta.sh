#!/usr/bin/env bash
# 建置 CUDD + OpenSTA（在 WSL2 Ubuntu 內以 root 執行）
set -euo pipefail

EDA=/root/eda
mkdir -p "$EDA"
cd "$EDA"

echo "########## 1/4 clone CUDD ##########"
if [ ! -d "$EDA/cudd" ]; then
  git clone --quiet https://github.com/cuddorg/cudd.git
fi
cd "$EDA/cudd"
git checkout --quiet 3.0.0

echo "########## 2/4 build+install CUDD ##########"
if [ ! -f "$EDA/cudd-install/lib/libcudd.a" ] && [ ! -f "$EDA/cudd-install/lib/libcudd.so" ]; then
  # git checkout 的時間戳會誤觸 autotools maintainer-mode，先用現有 automake 重新產生設定檔
  export DEBIAN_FRONTEND=noninteractive
  command -v automake >/dev/null 2>&1 || apt-get install -y automake autoconf libtool m4 >/tmp/autotools.log 2>&1
  autoreconf -fi >/tmp/cudd_autoreconf.log 2>&1 || { tail -30 /tmp/cudd_autoreconf.log; exit 1; }
  ./configure --prefix="$EDA/cudd-install" --enable-shared --enable-obj >/tmp/cudd_cfg.log 2>&1 || { tail -20 /tmp/cudd_cfg.log; exit 1; }
  make -j"$(nproc)" >/tmp/cudd_make.log 2>&1 || { tail -30 /tmp/cudd_make.log; exit 1; }
  make install >/tmp/cudd_install.log 2>&1 || { tail -20 /tmp/cudd_install.log; exit 1; }
fi
echo "CUDD include:"; ls "$EDA/cudd-install/include" | head
echo "CUDD lib:";     ls "$EDA/cudd-install/lib" | head

echo "########## 3/4 clone OpenSTA ##########"
cd "$EDA"
if [ ! -d "$EDA/OpenSTA" ]; then
  git clone --quiet https://github.com/parallaxsw/OpenSTA.git
fi
cd "$EDA/OpenSTA"
echo "OpenSTA HEAD: $(git log -1 --oneline)"

echo "########## 4/4 cmake build OpenSTA ##########"
mkdir -p build && cd build
cmake -DCUDD_DIR="$EDA/cudd-install" .. >/tmp/sta_cmake.log 2>&1 || { tail -40 /tmp/sta_cmake.log; exit 1; }
make -j"$(nproc)" >/tmp/sta_make.log 2>&1 || { tail -60 /tmp/sta_make.log; exit 1; }

echo "########## DONE ##########"
ls -la "$EDA/OpenSTA/build/sta"
echo "ALL_BUILD_DONE"

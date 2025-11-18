#!/bin/bash

git config --global --unset-all http.proxy
git config --global --unset-all https.proxy
git config --global --unset-all http.sslverify

IP=x.x.x.x
PROXY_URL="http://p_atlas:proxy%40123@${IP}:8888"

export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export no_proxy="127.0.0.1,.xx.com,localhost,local,.local"

git config --global http.proxy "${PROXY_URL}"
git config --global https.proxy "${PROXY_URL}"
git config --global http.sslverify false
git config --global https.sslverify false
git config --global http.sslVerify false
git config --global https.sslVerify false
# git config --global http.postBuffer 524288000
# git config --global http.lowSpeedLimit 0
# git config --global http.lowSpeedTime 999999
export GIT_SSL_NO_VERIFY=1

# 设置主镜像源（阿里云）
pip3 config set global.index-url "https://mirrors.aliyun.com/pypi/simple/"

# 添加备用镜像源（华为云）
pip3 config set global.extra-index-url "https://mirrors.huaweicloud.com/ascend/repos/pypi/"

# 配置信任的主机域名（需包含所有镜像域名）
pip3 config set install.trusted-host "mirrors.aliyun.com"
pip3 config set install.trusted-host "mirrors.huaweicloud.com"

curl -I www.baidu.com | grep HTTP

git config --global user.email "qiushixu@usc.edu"
git config --global user.name "Apocalypse990923-qshi"

# export PIP_TRUSTED_HOST=mirrors.aliyun.com
# export PIP_NO_VERIFY_CERTS=1
# export PYTHONHTTPSVERIFY=0


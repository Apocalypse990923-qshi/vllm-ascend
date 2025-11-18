ps -ef | grep "python" | grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {}
pkill -9 vllm
pkill -9 python

ps -ef | grep "python"| grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {};pkill -9 python; pkill -9 torchrun;ray stop
# É±Ä¬ÈÏ½ø³Ì
ps -ef | grep "defunct"|grep python| awk '{print $3}'|xargs -t -i kill -9 {};ps -ef | grep "defunct"|grep torchrun| awk '{print $3}'|xargs -t -i kill -9 {}


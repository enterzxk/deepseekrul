
pkill -9 -f "[o]3.sh"
pkill -9 -f "[r]un_candidate"
sleep 4
echo "remaining: $(pgrep -c -f '[r]un_candidate') scripts: $(pgrep -fc '[o]3.sh')"

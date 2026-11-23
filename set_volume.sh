for card in $(awk '/USB/ {print $1}' /proc/asound/cards); do
    amixer -c "$card" sset Master 90% 2>/dev/null
    amixer -c "$card" sset Speaker 90% 2>/dev/null
    amixer -c "$card" sset Headphone 90% 2>/dev/null
done

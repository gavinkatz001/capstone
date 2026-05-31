#! /bin/sh

# Only compile the fallyx_daemon which is used by the gateway
# Added -lpthread for threading support to fix BLE data stream choppiness
# Added cJSON.c for JSON processing support
gcc fallyx_daemon.c btlib.c cJSON.c -o fallyx_daemon -lpthread

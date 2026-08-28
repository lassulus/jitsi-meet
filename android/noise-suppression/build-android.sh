#!/usr/bin/env bash
# Cross-compiles libdfjni.so (DeepFilterNet3 noise suppression engine) for
# all Android ABIs the app ships, placing the results in the SDK's jniLibs
# directory, from where they are packaged into the AAR / APK.
#
# Requirements: rustup with the Android targets installed, cargo-ndk, and
# ANDROID_NDK_ROOT (or ANDROID_NDK_HOME) pointing at an NDK r27 install.
#
#   rustup target add aarch64-linux-android armv7-linux-androideabi \
#       i686-linux-android x86_64-linux-android

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}" ]; then
    echo "error: set ANDROID_NDK_ROOT or ANDROID_NDK_HOME" >&2
    exit 1
fi

out="../sdk/src/main/jniLibs"

cargo ndk \
    -t armeabi-v7a \
    -t arm64-v8a \
    -t x86 \
    -t x86_64 \
    -o "$out" \
    build --release "$@"

echo "Installed:"
find "$out" -name 'libdfjni.so' -exec ls -l {} \;

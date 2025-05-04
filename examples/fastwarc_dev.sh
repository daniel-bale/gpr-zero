mkdir -p ~/fastwarc-deps
cd ~/fastwarc-deps

# download the .deb files
apt-get download \
  zlib1g-dev \
  liblz4-dev 

# extract them into ./usr/
for pkg in *.deb; do
  dpkg-deb -x $pkg . 
done

# point the compiler at ./usr/include and ./usr/lib
export CPATH=$PWD/usr/include
export LIBRARY_PATH=$PWD/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$PWD/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
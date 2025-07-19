import os

SERVER_RELEASE="https://github.com/rcsoccersim/rcssserver/releases/download/rcssserver-19.0.0/rcssserver-19.0.0.tar.gz"
MONITOR_RELEASE="https://github.com/rcsoccersim/rcssmonitor/releases/download/rcssmonitor-19.0.1/rcssmonitor-19.0.1.tar.gz"

# assuming the system is Ubuntu 22.04 or similar
SERVER_DEPENDENCIES = "build-essential automake autoconf libtool flex bison libboost-all-dev"
MONITOR_DEPENDENCIES = "qtbase5-dev qt5-qmake libfontconfig1-dev libaudio-dev libxt-dev libglib2.0-dev libxi-dev libxrender-dev"


def install_dependencies():
    os.system("sudo apt update")
    os.system("sudo apt install -y %s" % SERVER_DEPENDENCIES)
    os.system("sudo apt install -y %s" % MONITOR_DEPENDENCIES)


def setup_release(release_url):
    release_name = os.path.basename(release_url)
    dir_name = release_name.replace(".tar.gz", "")

    os.makedirs("Downloads", exist_ok=True)
    os.chdir("Downloads")

    os.system("curl -LO %s" % release_url)
    os.system("tar zxvf %s > /dev/null" % release_name)

    os.chdir(dir_name)
    os.system("./configure")
    os.system("make")
    os.system("sudo make install")
    os.chdir("../..")


def update_rcfiles():
    os.system("echo 'PATH=\"/usr/local/bin:$PATH\"' >> ~/.profile")
    os.system("echo 'LD_LIBRARY_PATH=\"/usr/local/lib:$LD_LIBRARY_PATH\"' >> ~/.bashrc")
    os.system("echo 'export LD_LIBRARY_PATH' >> ~/.bashrc")
    os.system("echo 'chmod 0700 /run/user/1000/' >> ~/.bashrc")


def main():
    install_dependencies()
    setup_release(SERVER_RELEASE)
    setup_release(MONITOR_RELEASE)
    update_rcfiles()


if __name__ == "__main__":
    main()

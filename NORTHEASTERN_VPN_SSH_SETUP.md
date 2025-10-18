# Northeastern University HPC - VPN & SSH Setup Guide

## 🚀 Quick Connect (Each Time)

### Step 1: Connect to VPN

```bash
# Navigate to gp-saml-gui directory
cd ~/gp-saml-gui

# Run the SAML authentication tool
gp-saml-gui --portal vpn.northeastern.edu
```

This will:
- Open a browser window
- Prompt you to login with Microsoft (Northeastern SSO)
- Ask for your username: `mubashir.m`
- Ask for your password
- Complete Duo MFA authentication

### Step 2: Connect OpenConnect

After successful authentication, the tool will output a command like this:

```bash
echo [COOKIE_STRING] |
    sudo openconnect --protocol=gp '--useragent=PAN GlobalProtect' \
    --user=mubashir.m --os=linux-64 --usergroup=portal:prelogin-cookie \
    --passwd-on-stdin vpn.northeastern.edu
```

**Copy and paste this entire command into your terminal and run it.**

⚠️ **Keep this terminal window open!** The VPN connection stays active as long as this command is running.

### Step 3: SSH to HPC (in a NEW terminal)

Open a new terminal tab/window and run:

```bash
ssh mubashir.m@login.explorer.northeastern.edu
```

✅ **You're connected!**

---

## 📋 Important Addresses

| Service | Address |
|---------|---------|
| VPN Portal | `vpn.northeastern.edu` |
| SSH Login Node | `login.explorer.northeastern.edu` |
| OnDemand Portal | `https://ood.explorer.northeastern.edu/` |
| Compute Node (from VNC) | `c2185` |

---

## 🔌 Disconnect VPN

To disconnect from VPN:

1. Go to the terminal where `openconnect` is running
2. Press **Ctrl+C**
3. VPN will disconnect

---

## 📂 File Transfer Commands

### Download from HPC to Local

```bash
# Single file
scp mubashir.m@login.explorer.northeastern.edu:/path/to/file.txt ~/Downloads/

# Entire directory
scp -r mubashir.m@login.explorer.northeastern.edu:/path/to/folder/ ~/Downloads/

# Using rsync (recommended for large transfers)
rsync -avz mubashir.m@login.explorer.northeastern.edu:/path/to/folder/ ~/local-folder/
```

### Upload from Local to HPC

```bash
# Single file
scp ~/local/file.txt mubashir.m@login.explorer.northeastern.edu:~/

# Entire directory
scp -r ~/local/folder/ mubashir.m@login.explorer.northeastern.edu:~/destination/

# Using rsync
rsync -avz ~/local/folder/ mubashir.m@login.explorer.northeastern.edu:~/destination/
```

---

## 🔧 One-Time Initial Setup (Already Completed)

This section documents what was done during initial setup. **You don't need to do this again.**

### 1. Install GlobalProtect VPN Client

```bash
cd ~/Downloads
# Download PanGPLinux-5.2.3-c10.tgz from vpn.northeastern.edu
tar -xvf PanGPLinux-5.2.3-c10.tgz

# Install UI version (required dependencies)
sudo apt install libqt5webkit5
sudo dpkg -i GlobalProtect_UI_deb-5.2.3.0-10.deb
```

### 2. Install gp-saml-gui (SAML Authentication Helper)

```bash
# Install dependencies
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1 openconnect

# Clone the tool
cd ~
git clone https://github.com/dlenski/gp-saml-gui.git
cd gp-saml-gui

# Install it
pip3 install .
```

### 3. Set Default Browser

```bash
xdg-settings set default-web-browser google-chrome.desktop
```

---

## 🧪 Troubleshooting

### VPN Connected but SSH Fails

Check if VPN is truly connected:

```bash
# Check VPN interface
ip addr show | grep tun

# Check DNS resolution
nslookup login.explorer.northeastern.edu

# Check GlobalProtect status
globalprotect show --status
```

### DNS Not Resolving

Check your DNS settings:

```bash
cat /etc/resolv.conf
```

Should include Northeastern DNS servers (like `155.33.2.1`).

### SAML Authentication Cookie Expired

If you get authentication errors, the cookie has expired (they typically last 24 hours). Simply:

1. Disconnect VPN (Ctrl+C in openconnect terminal)
2. Run `gp-saml-gui --portal vpn.northeastern.edu` again
3. Re-authenticate
4. Use the new openconnect command

### Can't Find gp-saml-gui Command

Make sure you're in the right directory:

```bash
cd ~/gp-saml-gui
gp-saml-gui --portal vpn.northeastern.edu
```

Or check if it's in your PATH:

```bash
which gp-saml-gui
```

---

## 💡 Alternative Access Methods

### 1. OnDemand Web Portal (No VPN Required)

Access the HPC directly through your browser:

**URL:** `https://ood.explorer.northeastern.edu/`

Features:
- **Desktop VNC:** Visual desktop environment
- **Web Shell:** Terminal access through browser
- **File Browser:** Upload/download files via web interface
- **Job Management:** Submit and monitor jobs

### 2. VS Code Remote SSH (After VPN Connected)

With VPN active, you can use VS Code Remote SSH:

1. Install **Remote - SSH** extension in VS Code
2. Press `F1` → "Remote-SSH: Connect to Host"
3. Enter: `mubashir.m@login.explorer.northeastern.edu`
4. Browse and edit files directly on the cluster

---

## 📊 Typical Workflow

### For Development:

1. **Edit code locally** in your IDE
2. **Push to GitHub** from local machine
3. **Connect via VPN + SSH** to HPC
4. **Pull changes** on HPC: `git pull`
5. **Run training/jobs** on HPC
6. **Download results** via `scp` or OnDemand Files

### For Quick Tasks:

1. Use **OnDemand Web Shell** (no VPN needed)
2. Run commands directly in browser terminal
3. Download files via OnDemand File Browser

---

## 🔐 SSH Key Setup (Optional but Recommended)

To avoid typing your password every time:

### On Local Machine:

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@northeastern.edu"

# Copy public key to HPC
ssh-copy-id mubashir.m@login.explorer.northeastern.edu
```

### Or Manually:

```bash
# On local machine, get your public key
cat ~/.ssh/id_ed25519.pub

# Copy the output, then SSH to HPC and run:
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Now you can SSH without entering your password!

---

## 📞 Support

- **Northeastern IT Help:** help@northeastern.edu
- **Research Computing:** rc-help@northeastern.edu
- **OnDemand Portal:** https://ood.explorer.northeastern.edu/

---

## ✅ Summary

**Every time you need HPC access:**

1. Run `gp-saml-gui --portal vpn.northeastern.edu` → Login via browser
2. Copy and run the `openconnect` command it provides → Keep running
3. In new terminal: `ssh mubashir.m@login.explorer.northeastern.edu`
4. Work on the cluster
5. Press Ctrl+C in VPN terminal to disconnect when done

**That's it!** 🎉


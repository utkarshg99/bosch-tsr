# Installation guide for the UI

The documentation assumes that the user has python3 installed and python scripts can be excuted by running ```python3 /path/to/script/```

Intall Node.JS using binaries for Ubuntu from:

```
# Using Ubuntu
curl -fsSL https://deb.nodesource.com/setup_15.x | sudo -E bash -
sudo apt-get install -y nodejs

# Using Debian, as root
curl -fsSL https://deb.nodesource.com/setup_15.x | bash -
apt-get install -y nodejs
```

Clone the Repository using [Skip this step if you already have the Repositories]:
```
git clone /url/of/the/repo
git clone /url/of/the/repo
```

Install python dependencies by running (in the ```Backend``` folder):
```
python3 -m pip install -r requirements.txt
```

Install node dependencies by running (in the ```Frontend``` folder):
```
npm install
```

Run the backend server by running (in the ```Backend``` folder):
```
flask run
```

Now navigate to:
```
http://localhost:8080/
```
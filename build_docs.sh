rm -r docs
make html
mv build/html docs
touch docs/.nojekyll
MONGODB_USERNAME=$(gcloud secrets versions access 1 --secret=MONGODB_USERNAME) && export MONGODB_USERNAME
MONGODB_PASSWORD=$(gcloud secrets versions access 1 --secret=MONGODB_PASSWORD) && export MONGODB_PASSWORD
MONGODB_URI=$(gcloud secrets versions access 1 --secret=MONGODB_URI) && export MONGODB_URI
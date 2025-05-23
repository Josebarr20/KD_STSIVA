import wandb

# Set your API key
wandb.login()

# Set the source and destination projects
src_entity = "josebarr20"
src_project = "Classification_CIFAR10_Baseline_pilot"
dst_entity = "josebarr20"
dst_project = "KL_TEMP"

# Initialize the wandb API
api = wandb.Api()

# Get the runs from the source project
runs = api.runs(f"{src_entity}/{src_project}")

# Iterate through the runs and copy them to the destination project

for run in runs:
    # Get the run history and files
    if run.id == 'w7gt7ywj':
        history = run.history()
        files = run.files()

        # Create a new run in the destination project
        wandb.login(key="cd514a4398a98306cdedf0ffb4ed08532e9734e5")
        new_run = wandb.init(project=dst_project, entity=dst_entity, config=run.config, name=run.name,resume="allow")
        
        # Log the history to the new run
        for index, row in history.iterrows():
            new_run.log(row.to_dict())

        # Upload the files to the new run
        for file in files:
            file.download(replace=True)
            new_run.save(file.name,policy = "now")

        # Finish the new run
        new_run.finish()
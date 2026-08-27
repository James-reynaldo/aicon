from pathlib import Path

folder = Path("/scratch/aldo/aicon/src/aicon/drawer_tutorial/data/bad_prior/")  # <-- change this
log_folder = Path("/scratch/aldo/aicon/src/aicon/drawer_tutorial/Pairwise_Sweep/logs_bp")  # <-- change this

indices = {20105, 20013, 20011, 20078, 20012, 20040, 
           20038, 20165, 20072, 20043, 20047, 20099, 
           20077, 20051, 20055, 20080, 20046, 20045, 
           20036, 20069, 20091, 20052, 20014, 20092, 
           20239, 20006, 20100, 20044, 20017, 20056, 
           20053, 20008, 20037, 20015, 20058, 20180, 
           20033, 20093, 20097, 20031, 20067, 
           20050, 20009, 20071, 20240, 20114, 20054, 
           20074, 20076, 20079, 20020, 20194, 20073, 
           20060, 20035, 20010, 20048, 20041, 20049, 
           20028, 20164, 20068, 20104, 20032, 20042, 
           20070, 20034, 20007, 20075, 20059, 20234, 
           20027, 20030, 20098, 20096, 20057, 20029, 
           20039, 20016, 20095, 20018, 20019, 20094, 
           20129}

deleted = 0

# for f in folder.glob("experiment_store_job_*.db"):
#     try:
#         # Extract the 5-digit index
#         idx = int(f.name.split("_")[3])
#     except (IndexError, ValueError):
#         continue

#     if idx in indices:
#         print(f"Deleting {f.name}")
#         f.unlink()
#         deleted += 1

# print(f"\nDeleted {deleted} files.")

for log in log_folder.glob("1656266_*.out"):
    # filename like 1650901_97.out
    job = int(log.stem.split("_")[1])
    experiment_index = 20000 + job if job < 1000 else job

    if experiment_index in indices:
        print(f"Deleting {log.name}")
        log.unlink()
        deleted += 1

print(f"Deleted {deleted} log files.")
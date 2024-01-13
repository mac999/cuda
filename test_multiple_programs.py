import subprocess

if __name__ == "__main__":
    programs = ["python program1.py", "python program2.py", "python program3.py", ...]

    processes = [subprocess.Popen(command, shell=True) for command in programs]
    for process in processes:
        process.wait()

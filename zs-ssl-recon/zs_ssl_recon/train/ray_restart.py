import subprocess
import time


def main():
    while True:
        try:
            # Run the second script
            subprocess.run(["python", "-m", "zs_ssl_recon.train.ray"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Script crashed with exit code {e.returncode}. Restarting...")
            time.sleep(1)  # Optional: wait for a second before restarting
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            break


if __name__ == "__main__":
    main()

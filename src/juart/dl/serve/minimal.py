import ray


# Do not schedule the main function on the head node
@ray.remote
def main():
    print(f"Running main on {ray.get_runtime_context().get_node_id()}")

    from .inference import main

    main()


if __name__ == "__main__":
    print("__main__")

    results = main.options(num_cpus=1).remote()
    ray.get(results)

import json
import sys
from itertools import product
from typing import Annotated, Optional

import boto3
import typer
from dep_tools.aws import object_exists
from dep_tools.grids import get_tiles
from dep_tools.namers import S3ItemPath


def main(
    years: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    regions: Optional[str] = "ALL",
    limit: Optional[int] = None,
    dataset_id: str = "ammi",
    overwrite: Optional[bool] = False,
    bucket: str = "dep-public-staging",
) -> None:
    country_codes = None if regions.upper() == "ALL" else regions.split(",")

    tiles = get_tiles(country_codes=country_codes)
    tiles = list(tiles)

    if limit is not None:
        limit = int(limit)

    years = years.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{years} is not a valid value for --years")

    tasks = [
        {
            "tile-id": ",".join([str(i) for i in tile[0]]),
            "year": year,
            "version": version,
        }
        for tile, year in product(tiles, years)
    ]

    # If we don't want to overwrite, then we should only run tasks that don't already exist
    # i.e., they failed in the past or they're missing for some other reason
    if not overwrite:
        valid_tasks = []
        client = boto3.client("s3")
        for task in tasks:
            itempath = S3ItemPath(
                bucket=bucket,
                sensor="s2",
                dataset_id=dataset_id,
                version=version,
                time=task["year"],
            )
            stac_path = itempath.stac_path(task["tile-id"].split(","))

            if not object_exists(bucket, stac_path, client=client):
                valid_tasks.append(task)
            if len(valid_tasks) == limit:
                break
        # Switch to this list of tasks, which has been filtered
        tasks = valid_tasks
    else:
        # If we are overwriting, we just keep going
        pass

    if limit is not None:
        tasks = tasks[0:limit]

    json.dump(tasks, sys.stdout)


if __name__ == "__main__":
    typer.run(main)

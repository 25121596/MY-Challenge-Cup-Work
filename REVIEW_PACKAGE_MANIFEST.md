# Edge-Cloud Review Snapshot

This snapshot is for source and protocol review. It contains the active edge
protocol, cloud admission code, GUI-facing edge code, Integration runtime,
Scheduler training source, tests, configurations, standard inputs, processed
datasets, and current JSON result samples.

It excludes model weights, virtual environments, build products, caches, logs,
and raw source datasets larger than 20 MiB. The omissions are intentional:
they are not required to assess protocol semantics, routing behaviour, or test
coverage.

The asset-routing configuration intentionally has an empty authoritative
registry. A reviewer should assess the contract and decision trace first; no
device is pre-classified as important in this snapshot.

import { analyzeParcels, type ParcelFilter } from "./parcels";
type Request = {
    kind: "load";
    values: Float32Array;
    shape: number[];
} | {
    kind: "analyze";
    filter: ParcelFilter;
};
const scope = self as unknown as {
    onmessage: (event: MessageEvent<Request>) => void;
    postMessage: (value: unknown) => void;
    close: () => void;
};
let dataset: {
    values: Float32Array;
    shape: number[];
} | null = null;
scope.onmessage = ({ data }) => {
    try {
        if (data.kind === "load") {
            if (data.values.byteLength > 64 * 1024 * 1024)
                throw new Error("Parcel dataset exceeds the 64 MiB residency limit");
            dataset = data;
            scope.postMessage({ ok: true });
        }
        else {
            if (!dataset)
                throw new Error("Parcel dataset is unavailable");
            scope.postMessage({ ok: true, result: analyzeParcels(dataset.values, dataset.shape, data.filter) });
        }
    }
    catch (error) {
        scope.postMessage({ ok: false, error: error instanceof Error ? error.message : String(error) });
        scope.close();
    }
};

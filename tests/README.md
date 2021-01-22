# Testing

## Environment variables

Some tests require a test data_ file and source the pertinent information from
environment variables.
When running tests you should have, at a minimum, the following environment
variables defined:

- `TEST_FILE`: Path (we recommend a full path on principle) to the test data_
  file containing the raw data_ from your recording.
- `TEST_NCHANNELS`: The *total* number of channels in the recording living in
  `TEST_FILE`.
- `TEST_NACTIVE`: The number of *active* channels in the recording living in
  `TEST_FILE`.
- `TEST_NGROUPS`: A count of channel groups. In the testing, what channels get
  grouped together isn't particularly important, but you should make sure that
  `TEST_NGROUPS` evenly divides `TEST_NACTIVE`.
- `TEST_SRATE_HZ`: The sampling rate (in Hz) of the recording living in
  `TEST_FILE`.
- `TEST_NFRAMES`: The expected number of frames in the recording (i.e., the
  total number of samples divided by the total number of channels). For
  example, at 30000 Hz the number of frames in a one-second recording should be
  30000, irrespective of how many channels there are.

## Test file format

Right now the data_ is expected to be short int (16-bit signed integer), laid
out in column-major order, as it would be in acquisition, i.e., a sample per
channel for each consecutive timestep.
This is the format followed by, e.g., SpikeGLX.


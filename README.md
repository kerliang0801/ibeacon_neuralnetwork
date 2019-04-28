# ibeacon_neuralnetwork
work in progrss


## Install dependencies
To sniff bluetooth packets on ubuntu
```
sudo apt-get install bluez-hcidump
```

## Data Collection
In the folder - data collection, there are two scripts:

#### ibeacon_scan:

This script reads all available packets at real time:

To run, 
```
./ibeacon_scan    #this outputs the format as follows : "UUID: $UUID MAJOR: $MAJOR MINOR: $MINOR POWER: $POWER RSSI: $RSSI"
or 
./ibeacon_scan -b   #this output the formats as follows: "$UUID $MAJOR $MINOR $POWER $RSSI"
or
./ibeacon_scan -readestimote  #this only reads the four predefined UUID of the estimote beacons
```

or one can enter the 2nd argument as follows, to only read the packets of the specific beacon:
-coconut
-icy
-mint
-blueberry

#### waitall_scan

This script sniffs for four packets from each beacon - coconut, icy, mint, blueberry before making another sniff/
```
./waitall_scan  #waits for all four packets, outputs array of ID, corresponding RSSI and timestamp and writes to data.txt
```
to create a csv file, run ./waitall_scan
```
./waitall_scan -csv # name_of_file
```
"#" should represent the position where u are gathering the data

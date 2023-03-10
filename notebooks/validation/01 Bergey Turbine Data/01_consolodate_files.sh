#!/bin/bash
turbines="t007 t024 t028 t034 t041 t074 t083 t114 t133 t135 t139 t140 t169 t170 t182 t183 t192 t207 t221"
for i in $turbines;do
  echo $i
  if [ ! -d $i ];then
    mkdir $i
    mv *.$i.txt $i/
  fi
  pushd $i
  #echo "packet_date,sequenceNumber,system_state,last_fault,user_state,autorun_enabled,autostart_count,bus_voltage,ac_voltage,dc_current,dc_voltage,ac_frequency,output_power,energy_produced,soft_grid,aio_dsp_rev,wireless_last_operation,wireless_last_register,wireless_last_result" > $i.txt
  for j in $(ls *.txt);do
    tail -n +2 $j | grep "\S" >> $i.txt
  done
  mv $i.txt ..
  popd
done


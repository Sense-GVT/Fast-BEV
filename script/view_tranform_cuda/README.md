# View Transformation Latency on device




## Prepare

+ install `cmake-3.25.0-linux-x86_64` 
+ Use `check.ipynb` to generate input data with different shape


## Run script
```
cd cpp
sh recmake # cmake 
sh run.sh  # run script
```


## Result


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Device</th>
<th valign="center">number</th>
<th valign="center">2D Feature Size</th>
<th valign="center">3D Voxel Size</th>
<th valign="center">Latecy</th>

<tr>
<td align="center">CPU</td>
<td align="center">S1</td>
<td align="center">80x16x44</td>
<td align="center">80x128x128</td>
<td align="center">3.0ms</td>

</tr>

<tr>
<td align="center">CPU</td>
<td align="center">S2</td>
<td align="center">160x16x44</td>
<td align="center">160x128x128</td>
<td align="center">6.0ms</td>
</tr>
 
 <tr>
<td align="center">CPU</td>
<td align="center">S3</td>
<td align="center">80x32x88</td>
<td align="center">80x128x128</td>
<td align="center">3.3ms</td>
</tr>
 
<tr>
<td align="center">CPU</td>
<td align="center">S4</td>
<td align="center">160x32x88</td>
<td align="center">160x128x128</td>
<td align="center">6.6ms</td>
</tr>

<tr>
<td align="center">CPU</td>
<td align="center">S5</td>
<td align="center">80x32x88</td>
<td align="center">80x256x256</td>
<td align="center">12.3ms</td>
</tr>

<tr>
<td align="center">CPU</td>
<td align="center">S6</td>
<td align="center">160x32x88</td>
<td align="center">160x256x256</td>
<td align="center">24.1ms</td>
</tr>

<tr>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
</tr>

<tr>
<td align="center">GPU</td>
<td align="center">S1</td>
<td align="center">80x16x44</td>
<td align="center">80x128x128</td>
<td align="center">0.0059ms</td>

</tr>

<tr>
<td align="center">GPU</td>
<td align="center">S2</td>
<td align="center">160x16x44</td>
<td align="center">160x128x128</td>
<td align="center">0.019ms</td>
</tr>
 
 <tr>
<td align="center">GPU</td>
<td align="center">S3</td>
<td align="center">80x32x88</td>
<td align="center">80x128x128</td>
<td align="center">0.0062ms</td>
</tr>
 
<tr>
<td align="center">GPU</td>
<td align="center">S4</td>
<td align="center">160x32x88</td>
<td align="center">160x128x128</td>
<td align="center">0.019ms</td>
</tr>

<tr>
<td align="center">GPU</td>
<td align="center">S5</td>
<td align="center">80x32x88</td>
<td align="center">80x256x256</td>
<td align="center">0.024ms</td>
</tr>

<tr>
<td align="center">GPU</td>
<td align="center">S6</td>
<td align="center">160x32x88</td>
<td align="center">160x256x256</td>
<td align="center">0.028ms</td>
</tr>


 </tbody></table>



<!-- <p align="center"><img src="docs/main_figure.jpg" alt="Declip framework" width="800"/></p> -->







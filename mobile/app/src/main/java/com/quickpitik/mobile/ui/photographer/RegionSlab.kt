package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.layout.*
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.data.remote.RegionDto
import com.quickpitik.mobile.ui.theme.*

@Composable
internal fun RegionSlab(
    regions: List<RegionDto>,
    regionCode: String,
    provinceCode: String,
    onRegionChange: (String) -> Unit,
    onProvinceChange: (String) -> Unit,
    showValidation: Boolean = false,
    enabled: Boolean = true,
) {
    val selectedRegion = regions.firstOrNull { it.code == regionCode }
    val selectedProvince = selectedRegion?.provinces?.firstOrNull { it.code == provinceCode }
    var regionMenuExpanded by remember { mutableStateOf(false) }
    var provinceMenuExpanded by remember { mutableStateOf(false) }
    val isMissing = regionCode.isBlank()
    val isError = showValidation && isMissing

    Column {
        Kicker(
            text = if (isError) "Region  ·  FILL THIS SECTION" else "Region",
            color = if (isError) ErrorRed else SlateSoft
        )
        Spacer(modifier = Modifier.height(12.dp))

        DropdownField(
            label = "Region",
            valueLabel = selectedRegion?.name.orEmpty(),
            // The saved code can't resolve to a name until the list arrives,
            // so say it's loading rather than "Pick a region" — which would
            // read as "nothing is set" on a photographer who has set one.
            placeholder = if (regions.isEmpty()) "Loading regions…" else "Pick a region",
            expanded = regionMenuExpanded,
            onExpand = { regionMenuExpanded = true },
            enabled = enabled,
            onCollapse = { regionMenuExpanded = false },
        ) {
            regions.forEach { region ->
                DropdownMenuItem(
                    text = { Text(region.name, color = Ink, style = Typography.bodyMedium) },
                    onClick = {
                        onRegionChange(region.code)
                        regionMenuExpanded = false
                    },
                )
            }
        }

        if (selectedRegion != null) {
            Spacer(modifier = Modifier.height(12.dp))
            DropdownField(
                label = "Province",
                valueLabel = selectedProvince?.name.orEmpty(),
                placeholder = "Pick a province",
                expanded = provinceMenuExpanded,
                onExpand = { provinceMenuExpanded = true },
                enabled = enabled,
                onCollapse = { provinceMenuExpanded = false },
            ) {
                selectedRegion.provinces.forEach { province ->
                    DropdownMenuItem(
                        text = { Text(province.name, color = Ink, style = Typography.bodyMedium) },
                        onClick = {
                            onProvinceChange(province.code)
                            provinceMenuExpanded = false
                        },
                    )
                }
            }
        }
    }
}


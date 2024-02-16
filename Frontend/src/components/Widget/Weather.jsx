import React, { useState, useEffect } from 'react';
import axios from 'axios';
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";

export default function Weather() {
    const [weatherData, setWeatherData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(
                'https://api.open-meteo.com/v1/forecast?latitude=1.2897&longitude=103.8501&hourly=temperature_2m,soil_temperature_0cm&timezone=Asia%2FSingapore'
                );
                setWeatherData(response.data);
            } catch (error) {
                console.error('Error fetching weather data:', error);
            }
        };
    fetchData();
    }, []);

    const generateChartOptions = () => {
        if (!weatherData || !weatherData.hourly || !weatherData.hourly.time) return null;

    const categories = weatherData.hourly.time;
    const temperatureSeries = weatherData.hourly.temperature_2m;
    const soilTemperatureSeries = weatherData.hourly.soil_temperature_0cm;

    const uniqueDates = new Set();

        const options = {
            chart: {
                type: 'line',
            },
            title: {
                text: 'Weather Variables',
            },
            xAxis: {
                type: 'datetime',
                categories: categories,
                labels: {
                    formatter: function () {
                        // convert to date and format date-month
                        const date = new Date(this.value);
                        // still display full date for each points
                        const dateString = `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
                        // filter unique dates for label only
                        if (!uniqueDates.has(dateString)) {
                            uniqueDates.add(dateString);
                            // format date-month
                            return Highcharts.dateFormat('%d %b', date);
                        } else {
                            // fill with empty string
                            return '';
                        }
                    },
                },
            },
            yAxis: [
                {
                    title: {
                        text: 'Temperature (°C)',
                    },
                    labels: {
                        format: '{value} °C',
                    },
                },
                {
                    title: {
                        text: 'Soil Temperature (°C)',
                    },
                    labels: {
                        format: '{value} °C',
                    },
                    opposite: true,
                },
            ],
            series: [
                {
                    name: 'Temperature',
                    data: temperatureSeries,
                    yAxis: 0,
                },
                {
                    name: 'Soil Temperature',
                    data: soilTemperatureSeries,
                    yAxis: 1,
                },
            ],
        };  
            return options;
    };

  return (
    <div>
        {weatherData && (
            <HighchartsReact highcharts={Highcharts} options={generateChartOptions()} />
        )}
    </div>
  );
};
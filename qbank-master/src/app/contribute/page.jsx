'use client'
import React from 'react'
import { useEffect, useState } from 'react'
import axios from 'axios'
import { parseCookies, setCookie } from 'nookies'
export default function page() {
    const { email, expert } = parseCookies();
    useEffect(() => {
        if (!email) {
            window.location.href = '/Signin'
        }
        if (expert == '-1') {
            window.location.href = '/Exam'
        }

    }, [])

    const [question, setQuestion] = useState('')
    const [topic, setTopic] = useState('')
    const [Correct, setCorrect] = useState('')
    const handlesubmit = async (e) => {
        e.preventDefault()
        if (!question || !topic) {
            alert('Please fill all the fields')
            return
        }
        const apiUrl = 'https://api.languagetool.org/v2/check';
        const language = 'en-US';

        try {
            const response = await fetch(`${apiUrl}?language=${language}&text=${encodeURIComponent(question)}`);
            const data = await response.json();

            // Process the API response to extract the corrected text
            const correctedText = data.matches.map((match) => match.replacements[0].value).join(' ');
            //   setCorrectedText(correctedText);
            console.log(correctedText)
            setCorrect(correctedText)
            if (correctedText != '') {
                alert('The question is not grammatically correct. Suggestion: ' + correctedText + '  Please correct the question and submit again')
                return
            }
        } catch (error) {
            console.error('Error while calling the API:', error);
        }


        const res = await fetch('http://localhost:8080/qms/question/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                email: email,
                subject: 'Computer Science',
                topic: topic,
            })

        })
        const data = await res.json()
        console.log(data)
        setShowResult(true)
        setStatus(data)


    }
    const [showResult, setShowResult] = useState(false);
    const [status, setStatus] = useState();
    return (
        <div>
            <div className="min-h-screen p-6 bg-gray-100 flex  justify-center">
                <div className="container max-w-screen-lg m-5  mx-auto">
                    {showResult ? <div>
                        <div className="container mx-auto mt-4 p-4 items-center">
                            <div className="flex flex-col justify-center items-center">
                                {status.status ? (
                                    <>
                                        <h1 className="text-2xl font-semibold mb-4 text-center">Thank you for your contribution</h1><div className='flex flex-row'>
                                        <button onClick={() => setShowResult(false)} className="bg-black hover:bg-gray-700 text-white font-bold py-2 px-4 rounded mt-4">
                                            Contribute More
                                        </button>
                                        <button
                                            className="bg-black hover:bg-gray-700 text-white font-bold py-2 px-4 rounded mt-4 ml-4"
                                            onClick={() => window.location.href = '/'}
                                        >
                                            Go to Home
                                        </button>
                                        </div>
                                    </>
                                ) : (
                                    <>
                                        <h1 className="text-2xl text-red-300 font-semibold mb-4 text-center">{status.message}</h1>
                                        <button onClick={() => setShowResult(false)} className="bg-black hover:bg-gray-700 text-white font-bold py-2 px-4 rounded mt-4">
                                            Try Again
                                        </button>
                                    </>
                                )}
                            </div>

                            {/* <h1 className="text-2xl font-semibold mb-4 text-center">Thank you for your contribution</h1> */}
                        </div>
                    </div>
                        : <div>
                            {/* <h2 className="font-semibold text-xl text-gray-600">Question details</h2> */}
                            {/* <p className="text-gray-500 mb-6">Form is mobile responsive. Give it a try.</p> */}

                            <div className="bg-white rounded shadow-lg p-4 px-4 md:p-8 mb-6">
                                <div className="grid gap-4 gap-y-2 text-sm grid-cols-1 lg:grid-cols-3">
                                    <div className="text-gray-600">
                                        <p className="font-medium text-lg">Question Details</p>
                                        {/* <p>Please fill out all the fields.</p> */}
                                    </div>

                                    <div className="lg:col-span-2">
                                        <div className="grid gap-4 gap-y-2 text-sm grid-cols-1 md:grid-cols-5">
                                            <div className="md:col-span-5">
                                                <label htmlFor="full_name">Full Question</label>
                                                <textarea type="text" name="full_name" id="full_name" spellCheck="true" value={question} onChange={(e) => setQuestion(e.target.value)}
                                                    className="h-20 border mt-1 rounded px-4 w-full bg-gray-50" placeholder='Write the question here' />
                                                {/* {Correct!=''?<p className='text-grey-400'>Suggestion: {Correct}</p>:<></>} */}
                                            </div>

                                            <div className="md:col-span-5">
                                                <label htmlFor="email">Topic</label>
                                                <select className='h-10 border mt-1 rounded px-4 w-full bg-gray-50' value={topic} onChange={(e) => setTopic(e.target.value)}>
                                                    <option value="">Select</option>
                                                    {/* <option vaalue="Computer Programming">Computer programming</option> */}
                                                    <option value="Algorithms">Algorithms</option>
                                                    <option value="Data Structures">Data Structures</option>
                                                    <option value="Operating Systems">Operating Systems</option>
                                                    <option value="Networking">Networking</option>
                                                    <option value="Databases">Database Management Systems</option>
                                                    <option value="Computer Architecture">Computer Architecture</option>
                                                    <option value="Theory of Computation">Theory of Computation</option>
                                                </select>
                                            </div>


                                            <div className="md:col-span-5 text-right">
                                                <div className="inline-flex items-end">
                                                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                                                        onClick={handlesubmit}>
                                                        Submit</button>
                                                </div>
                                            </div>

                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>}
                </div>
            </div>
        </div>
    )
}
